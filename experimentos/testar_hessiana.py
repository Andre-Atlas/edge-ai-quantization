"""Testa a hipotese do NaN do GPTQ no SmolLM3: a Hessiana (H = soma de x^T x) das camadas o_proj 0, 5 e 34 e
down_proj 10 e mal condicionada, e a fatoracao de Cholesky do mlx_lm.quant.gptq falha.

  python experimentos/testar_hessiana.py [--modelo HuggingFaceTB/SmolLM3-3B] [--ate 35]

Reproduz a coleta do gptq.py (Catcher, calibration_v5, sequencia 512, todas as amostras, lote 8) mas camada a
camada, com pesos carregados de forma preguicosa e liberados apos cada camada (pico de uns 4-5 GB, nao 22 GB).
Para cada Hessiana compara (a) a forma original (cada produto x^T x em bf16, somado em float32) com (b) a exata
(tudo em float32) e repete compute_inverse_hessian (amortecimento de 1% da media da diagonal + Cholesky).
Nao grava nada no projeto. Trava: aborta se a memoria disponivel cair abaixo de --piso-gb.
"""
import argparse
import os
import threading
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import psutil
from mlx.utils import tree_map
from mlx_lm.models.base import create_attention_mask
from mlx_lm.quant.utils import load_data
from mlx_lm.utils import load

GB = 1e9


class Captura(nn.Module):
    def __init__(self, modulo, guardar_h):
        super().__init__()
        self.modulo = modulo
        self.guardar_h = guardar_h
        self.h_orig = mx.array(0.0)
        self.h_exata = mx.array(0.0)
        self.maxabs = 0.0
        self.nao_finitos = 0

    def __call__(self, x, *a, **k):
        xf = x.flatten(0, -2)
        self.maxabs = max(self.maxabs, mx.abs(xf).max().item())
        self.nao_finitos += int((~mx.isfinite(xf)).sum().item())
        if self.guardar_h:
            self.h_orig = self.h_orig + xf.T @ xf                      # como o gptq.py do mlx-lm
            x32 = xf.astype(mx.float32)
            self.h_exata = self.h_exata + x32.T @ x32                  # referencia exata
            mx.eval(self.h_orig, self.h_exata)
        return self.modulo(x, *a, **k)


def inversa_como_gptq(H):
    """Copia de compute_inverse_hessian; devolve (damp, n_nan_chol, n_nan_hinv)."""
    with mx.stream(mx.cpu):
        H = mx.array(H)
        damp = 1e-2 * mx.mean(mx.diag(H))
        d = mx.arange(H.shape[0])
        H[d, d] += damp
        c = mx.linalg.cholesky(H)
        mx.eval(c)
        n1 = int(mx.isnan(c).sum().item())
        ci = mx.linalg.cholesky_inv(c)
        hinv = mx.linalg.cholesky(ci, upper=True)
        mx.eval(hinv)
        return damp.item(), n1, int(mx.isnan(hinv).sum().item())


def diagnosticar(nome, cap):
    out = [f"{nome}: max|x|={cap.maxabs:.4g} x_nao_finitos={cap.nao_finitos}"]
    for rot, H in (("H_orig ", cap.h_orig), ("H_exata", cap.h_exata)):
        H = H.astype(mx.float32)
        mx.eval(H)
        nf = int((~mx.isfinite(H)).sum().item())
        dg = mx.diag(H)
        dmax, dmin, dmed = dg.max().item(), dg.min().item(), mx.median(dg).item()
        if nf == 0:
            damp, n1, n2 = inversa_como_gptq(H)
            res = f"damp={damp:.3g} cond~max/damp={dmax / damp:.3g} NaN(chol)={n1} NaN(Hinv)={n2}"
            if H.shape[0] <= 4096:
                with mx.stream(mx.cpu):
                    sim = mx.abs(H - H.T).max().item()
                    ev = mx.linalg.eigvalsh(0.5 * (H + H.T)).min().item()
                res += f" | assimetria max={sim:.3g} autovalor_min={ev:.3g} (damp={damp:.3g})"
        else:
            res = "H com NaN/Inf: nao fatorado"
        out.append(f"   {rot} nao_finitos={nf} diag max={dmax:.3g} med={dmed:.3g} min={dmin:.3g} zeros={int((dg == 0).sum().item())} | {res}")
    return "\n".join(out)


def vigiar(piso_gb):
    def f():
        while True:
            if psutil.virtual_memory().available < piso_gb * GB:
                print(f"\n[ABORTADO] memoria disponivel < {piso_gb} GB", flush=True)
                os._exit(3)
            time.sleep(0.5)
    threading.Thread(target=f, daemon=True).start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="HuggingFaceTB/SmolLM3-3B")
    ap.add_argument("--ate", type=int, default=35, help="processa as camadas 0..ate-1")
    ap.add_argument("--down", default="9,10", help="camadas cujo down_proj tem a Hessiana coletada")
    ap.add_argument("--piso-gb", type=float, default=2.0)
    ap.add_argument("--cpu", action="store_true", help="roda tudo na CPU do MLX (evita disputar a GPU com outros jobs; mais lento)")
    a = ap.parse_args()
    vigiar(a.piso_gb)
    if a.cpu:
        mx.set_default_device(mx.cpu)
    mx.set_memory_limit(int(6 * GB))
    mx.set_cache_limit(int(0.5 * GB))
    down = None if a.down == "todas" else {int(x) for x in a.down.split(",")}

    model, tok = load(a.modelo, lazy=True)
    dados = load_data(tok, -1, 512)
    lotes = [dados[s:s + 8] for s in range(0, dados.shape[0], 8)]
    print(f"{a.modelo}: {dados.shape[0]} amostras de 512 tokens, {len(lotes)} lotes; embed dtype {model.model.embed_tokens.weight.dtype}", flush=True)
    acts = [model.model.embed_tokens(b) for b in lotes]
    mx.eval(acts)

    for i in range(a.ate):
        layer = model.model.layers[i]
        co = Captura(layer.self_attn.o_proj, True)
        cd = Captura(layer.mlp.down_proj, down is None or i in down)
        cq, cg = Captura(layer.self_attn.q_proj, True), Captura(layer.mlp.gate_proj, True)   # H de q/k/v e de gate/up
        layer.self_attn.o_proj, layer.mlp.down_proj, layer.self_attn.q_proj, layer.mlp.gate_proj = co, cd, cq, cg
        novas = []
        for h in acts:
            mask = create_attention_mask(h, None)
            novas.append(layer(h, mask, cache=None))
            mx.eval(novas[-1])
        acts = novas
        print(diagnosticar(f"camada {i:2d} o_proj ", co), flush=True)
        print(diagnosticar(f"camada {i:2d} q_proj ", cq), flush=True)
        print(diagnosticar(f"camada {i:2d} gate_proj", cg), flush=True)
        if down is None or i in down:
            print(diagnosticar(f"camada {i:2d} down_proj", cd), flush=True)
        saida_ruim = sum(int((~mx.isfinite(h)).sum().item()) for h in acts)
        if saida_ruim:
            print(f"   !! saida da camada {i} tem {saida_ruim} valores nao finitos", flush=True)
        layer.update(tree_map(lambda p: mx.zeros((1,), p.dtype), layer.parameters()))  # libera os pesos
        del co, cd, cq, cg
        mx.clear_cache()


if __name__ == "__main__":
    main()
