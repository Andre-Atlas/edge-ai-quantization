"""GPTQ camada a camada: a mesma quantizacao do mlx-lm 0.32.2 (mlx_lm.quant.gptq) com pico de memoria muito menor.

  python experimentos/gptq_camada.py --model Qwen/Qwen3-0.6B --mlx-path DESTINO [--bits 4] [--group-size 64] [--seed 0]
                                     [--hessiana bf16|fp32] [--piso-gb 2.5]

O gptq.py do mlx-lm instala um acumulador de Hessiana em TODAS as camadas lineares, roda o modelo e so depois quantiza
(Hessianas de todas as camadas vivas ao mesmo tempo: ~1,5 x seu tamanho, 14,7 GB no Qwen3-1,7B). Aqui o modelo e percorrido
camada a camada: coleta-se a Hessiana de uma camada, quantiza-se, solta-se, e as ATIVACOES PASSADAS A PROXIMA CAMADA SAO AS
DO MODELO ORIGINAL (nao as do quantizado), exatamente como no mlx-lm, onde todas as Hessianas vem de uma unica passada no
modelo original. Matematica identica; o laco de quantizacao (inversa da Hessiana, erro, atualizacao dos pesos, escalas por
grupo) e copiado do mlx-lm sem alteracoes. A semente, os dados de calibracao (calibration_v5, todas as amostras, 512 tokens),
o lote de 8 e as camadas de reserva (embedding a 6 bits, grupo 64) tambem sao os mesmos.

--hessiana bf16 (padrao): reproduz o Catcher do mlx-lm (produto x^T x em bf16, somado em float32) -> VALIDACAO contra o padrao.
--hessiana fp32: calcula o produto em float32 -> desvio A do protocolo-v2; evita o NaN do SmolLM3.
Apenas modelos sem janela deslizante (os cinco do estudo). Trava de seguranca: aborta se a memoria disponivel < --piso-gb.
"""
import argparse
import os
import threading
import time

import mlx.core as mx
import mlx.nn as nn
import psutil
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.switch_layers import SwitchLinear
from mlx_lm.quant.gptq import quantize  # empacotamento de bits do mlx-lm
from mlx_lm.quant.utils import load_data
from mlx_lm.utils import compute_bits_per_weight, load, save
from tqdm import tqdm

GB = 1e9
TIPOS_GPTQ = {nn.Linear, SwitchLinear}


class Captura(nn.Module):
    """Equivale ao Catcher do mlx-lm: acumula H = soma de x^T x nas entradas do modulo linear."""

    def __init__(self, modulo, modo):
        super().__init__()
        self.modulo = modulo
        self.modo = modo
        self.H = mx.array(0.0)

    def __call__(self, x, *args, **kwargs):
        xf = x.flatten(0, -2)
        if self.modo == "fp32":
            xf = xf.astype(mx.float32)
        self.H = self.H + xf.T @ xf
        return self.modulo(x, *args, **kwargs)


def inversa_da_hessiana(H):
    """Copia de compute_inverse_hessian (mlx_lm.quant.gptq)."""
    with mx.stream(mx.cpu):
        damp = 1e-2 * mx.mean(mx.diag(H))
        diag = mx.arange(H.shape[0])
        H[diag, diag] += damp
        H = mx.linalg.cholesky(H)
        H = mx.linalg.cholesky_inv(H)
        Hinv = mx.linalg.cholesky(H, upper=True)
        return Hinv


def fazer_erro(bits):
    n_bins = 2**bits - 1

    @mx.compile
    def gptq_error(w, d, scales, biases):  # copia de gptq_error (mlx_lm.quant.gptq)
        q = mx.clip(mx.round((w - biases) / scales), 0.0, n_bins)
        q = scales * q + biases
        return (w - q) / d

    return gptq_error


def quantizar_linear(lin, H, bits, group_size, erro):
    """Copia do laco de quantizacao de gptq_quantize para uma camada linear. Devolve (camada quantizada, NaN em Hinv)."""
    Hinv = inversa_da_hessiana(H)
    mx.eval(Hinv)
    n_nan = int(mx.isnan(Hinv).sum().item())
    orig_type = lin.weight.dtype
    W = lin.weight.astype(mx.float32)
    all_scales, all_biases = [], []
    for i in range(0, W.shape[-1], group_size):
        j = i + group_size
        Wl = W[..., i:j]
        err = mx.zeros_like(Wl)
        _, scales, biases = mx.quantize(Wl, bits=bits, group_size=group_size)
        all_scales.append(scales)
        all_biases.append(biases)
        for k in range(group_size):
            k += i
            w = W[..., k : k + 1]
            d = Hinv[k, k]
            e = erro(w, d, scales, biases)
            W[..., k:j] -= e @ Hinv[k : k + 1, k:j]
            err[..., k - i : k - i + 1] = e
            mx.eval(err, W)
        W[..., j:] -= err @ Hinv[i:j, j:]
    scales = mx.concatenate(all_scales, axis=-1)
    biases = mx.concatenate(all_biases, axis=-1)
    Wq = quantize(W, bits, scales, biases)
    camada = lin.to_quantized(bits=bits, group_size=group_size)
    camada.weight = Wq
    camada.scales = scales
    camada.biases = biases
    camada.set_dtype(orig_type)
    mx.eval(camada)
    return camada, n_nan


def vigiar(piso_gb):
    def f():
        while True:
            if psutil.virtual_memory().available < piso_gb * GB:
                print(f"\n[ABORTADO] memoria disponivel < {piso_gb} GB", flush=True)
                os._exit(3)
            time.sleep(0.5)

    threading.Thread(target=f, daemon=True).start()


def quantizar_camada_a_camada(model, lotes, bits, group_size, fallback_bits, fallback_group_size, modo, erro):
    interno = model.model
    acts = [interno.embed_tokens(b) for b in lotes]
    mx.eval(acts)
    total_nan = 0
    for idx, camada in enumerate(tqdm(interno.layers, desc="Camadas")):
        assert not getattr(camada, "use_sliding", False), "janela deslizante nao suportada"
        folhas = tree_flatten(camada.leaf_modules(), is_leaf=nn.Module.is_module)
        capt = [(k, Captura(m, modo)) for k, m in folhas if type(m) in TIPOS_GPTQ]
        camada.update_modules(tree_unflatten(capt))
        novas = []
        for h in acts:                                    # passada no modelo ORIGINAL (pesos ainda em precisao cheia)
            mask = create_attention_mask(h, None)
            novas.append(camada(h, mask, None))
            mx.eval(novas[-1], [c.H for _, c in capt])
        quant = []
        for k, c in capt:
            lin, n = quantizar_linear(c.modulo, c.H, bits, group_size, erro)
            total_nan += n
            if n:
                print(f"   !! camada {idx} {k}: Hinv com {n} NaN", flush=True)
            quant.append((k, lin))
            c.H = None
        camada.update_modules(tree_unflatten(quant))
        acts = novas                                      # ativacoes do modelo original para a proxima camada
        mx.clear_cache()
    if hasattr(model, "lm_head") and type(model.lm_head) in TIPOS_GPTQ:     # modelos sem embedding amarrado
        c = Captura(model.lm_head, modo)
        for h in acts:
            c(interno.norm(h))
            mx.eval(c.H)
        lin, n = quantizar_linear(c.modulo, c.H, bits, group_size, erro)
        total_nan += n
        model.lm_head = lin
    config = {"bits": bits, "group_size": group_size}
    reserva = {"bits": fallback_bits, "group_size": fallback_group_size}
    q = []
    for k, l in tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module):
        if hasattr(l, "to_quantized"):                    # embedding (e o que sobrar) como no mlx-lm
            config[k] = reserva
            q.append((k, l.to_quantized(**reserva)))
    if q:
        model.update_modules(tree_unflatten(q))
    return model, config, total_nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", "-m", required=True)
    ap.add_argument("--mlx-path", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--fallback-bits", type=int, default=6)
    ap.add_argument("--fallback-group-size", type=int, default=64)
    ap.add_argument("--num-samples", type=int, default=-1)
    ap.add_argument("--sequence-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--hessiana", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--piso-gb", type=float, default=2.5)
    a = ap.parse_args()
    vigiar(a.piso_gb)
    t0 = time.time()

    mx.random.seed(a.seed)                                # mesma ordem do mlx-lm: semente, carga do modelo, dados
    model, tok, config = load(a.model, lazy=True, return_config=True)
    dados = load_data(tok, a.num_samples, a.sequence_length)
    lotes = [dados[s : s + 8] for s in range(0, dados.shape[0], 8)]
    print(f"{a.model}: {dados.shape[0]} amostras, {len(lotes)} lotes, Hessiana {a.hessiana}", flush=True)

    model, config["quantization"], n_nan = quantizar_camada_a_camada(
        model, lotes, a.bits, a.group_size, a.fallback_bits, a.fallback_group_size, a.hessiana, fazer_erro(a.bits))
    print(f"Quantized model with {compute_bits_per_weight(model):.3f} bits per weight.", flush=True)
    save(a.mlx_path, a.model, model, tok, config)
    print(f"[camada_a_camada] pico MLX {mx.get_peak_memory() / GB:.2f} GB; NaN em Hinv: {n_nan}; {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
