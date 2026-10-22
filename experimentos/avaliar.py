"""Avalia um modelo MLX em duas fases independentes.

  --fase qualidade : perplexidade no WikiText-2 (teste) + acuracia zero-shot (lm-eval).
                     Nao depende de tempo, pode rodar em paralelo com outros jobs.
  --fase velocidade: prefill/decode (tokens/s) e pico de memoria.
                     Deve rodar SOZINHA na maquina (GPU e banda de memoria compartilhadas).

Uso: python experimentos/avaliar.py <modelo> --fase qualidade|velocidade [--nll-saida arq.npy] [--limite N]
Imprime uma unica linha JSON na ultima linha da saida.
"""
import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

TAREFAS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande"]


def nll_por_janela(model, tokenizer, seq_len=2048):
    """NLL media por token em cada janela de 2048 tokens do WikiText-2 (teste, raw).

    Uma janela por vez: com vocabulario de ~152k tokens, os logits em float32 de UMA janela
    ja ocupam ~1,2 GB; em lotes de 4 passavam de 12 GB por processo e esgotavam a GPU.
    """
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = mx.array(tokenizer.encode("\n\n".join(ds["text"])))
    n = (ids.size // seq_len) * seq_len
    blocos = ids[:n].reshape(-1, seq_len)
    por_janela = []
    for i in range(blocos.shape[0]):
        x = blocos[i : i + 1]
        logits = model(x[:, :-1]).astype(mx.float32)
        nll = mx.logsumexp(logits, axis=-1) - mx.take_along_axis(logits, x[:, 1:, None], axis=-1).squeeze(-1)
        por_janela.append(mx.mean(nll).item())
        del logits, nll
        mx.clear_cache()
    return np.array(por_janela), seq_len - 1


def acuracias(modelo, limite=None):
    """Acuracia zero-shot via `mlx_lm evaluate`; usa acc_norm quando existe, senao acc."""
    with tempfile.TemporaryDirectory() as tmp:
        # o lm-eval nomeia o arquivo de saida com o caminho do modelo: usa link curto para nao estourar o limite
        curto = Path(tmp) / "m"
        curto.symlink_to(Path(modelo).resolve()) if Path(modelo).exists() else None
        cmd = [sys.executable, "-m", "mlx_lm", "evaluate", "--model", str(curto) if curto.exists() else modelo, "--tasks", *TAREFAS,
               "--num-shots", "0", "--no-apply-chat-template", "--batch-size", "4", "--output-dir", tmp]
        if limite:
            cmd += ["--limit", str(limite)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("mlx_lm evaluate falhou:\n" + r.stderr[-2000:])
        saida = r.stdout
    dec, i, res = json.JSONDecoder(), 0, {}
    while (i := saida.find("{", i)) != -1:
        try:
            obj, fim = dec.raw_decode(saida, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict) and obj.get("name") in TAREFAS:
            usa = "acc_norm,none" if "acc_norm,none" in obj else "acc,none"
            res[obj["name"]] = (obj[usa], obj[usa.replace(",none", "_stderr,none")])
        i = fim
    faltam = set(TAREFAS) - set(res)
    if faltam:
        raise RuntimeError(f"tarefas sem resultado: {faltam}")
    return res


def velocidade(model, tokenizer, prompt_tokens=512, gen_tokens=128, trials=5):
    from mlx_lm import stream_generate

    prompt = mx.array(tokenizer.encode("hello " * prompt_tokens)[:prompt_tokens])
    pre, dec = [], []
    for t in range(trials + 1):  # a 1a rodada e aquecimento
        r = None
        for r in stream_generate(model, tokenizer, prompt, max_tokens=gen_tokens):
            pass
        if t > 0:
            pre.append(r.prompt_tps)
            dec.append(r.generation_tps)
    return pre, dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modelo")
    ap.add_argument("--fase", required=True, choices=["qualidade", "velocidade"])
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--nll-saida", help="salva a NLL por janela (.npy) para bootstrap")
    ap.add_argument("--limite", type=int, help="limita exemplos por tarefa (so para testes)")
    a = ap.parse_args()
    import gc
    import os

    from mlx_lm import load

    # Teto por processo = pesos + 4 GB (logits de uma janela e temporarios). Acima disso o MLX
    # libera cache em vez de pedir mais a GPU, protegendo o WindowServer e o resto do macOS.
    pesos_gb = sum(f.stat().st_size for f in Path(a.modelo).glob("*.safetensors")) / 1e9 if Path(a.modelo).is_dir() else 8
    limite_gb = float(os.environ.get("EAQ_LIMITE_GPU_GB", pesos_gb + 4))
    mx.set_memory_limit(int(limite_gb * 1e9))
    mx.set_cache_limit(int(1e9))

    out = {}
    media = lambda v: sum(v) / len(v)
    dp = lambda v: (sum((x - media(v)) ** 2 for x in v) / (len(v) - 1)) ** 0.5
    if a.fase == "qualidade":
        model, tok = load(a.modelo)
        mx.reset_peak_memory()
        nll, ntok_janela = nll_por_janela(model, tok, a.seq)
        if a.nll_saida:
            np.save(a.nll_saida, nll)
        out.update(ppl_wikitext2=math.exp(nll.mean()), janelas=len(nll), tokens_avaliados=len(nll) * ntok_janela, seq=a.seq,
                   pico_memoria_ppl_gb=mx.get_peak_memory() / 1e9)
        del model, tok
        gc.collect()
        mx.clear_cache()  # libera a GPU antes de o lm-eval carregar outra copia do modelo
        acc = acuracias(a.modelo, a.limite)
        for t, (v, e) in acc.items():
            out[f"acc_{t}"], out[f"acc_{t}_ep"] = v, e
        out["acc_media"] = media([v for v, _ in acc.values()])
    else:
        model, tok = load(a.modelo)
        mx.reset_peak_memory()
        pre, dec = velocidade(model, tok)
        out.update(prefill_tps=media(pre), prefill_tps_dp=dp(pre), decode_tps=media(dec), decode_tps_dp=dp(dec),
                   pico_memoria_gb=mx.get_peak_memory() / 1e9)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
