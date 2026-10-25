"""Compara um modelo quantizado pelo gptq_camada.py com o gerado pelo `mlx_lm gptq` (padrao).

  python experimentos/comparar_gptq.py NOVO PADRAO [--nll ARQ.npy]

1) tensor a tensor: pesos empacotados (uint32), escalas e biases; 2) opcional: NLL por janela do WikiText-2 do modelo NOVO contra o
ARQ.npy do padrao (resultados/nll/<ident>.npy). Se os tensores forem identicos, a perplexidade tambem e.
"""
import argparse
import glob
from pathlib import Path

import mlx.core as mx
import numpy as np


def carregar(pasta):
    t = {}
    for f in sorted(glob.glob(f"{pasta}/*.safetensors")):
        t.update(mx.load(f))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("novo")
    ap.add_argument("padrao")
    ap.add_argument("--nll")
    a = ap.parse_args()
    n, p = carregar(a.novo), carregar(a.padrao)
    print(f"tensores: novo {len(n)} | padrao {len(p)} | so em um dos dois: {sorted(set(n) ^ set(p))[:5]}")
    iguais, difs = 0, []
    for k in sorted(set(n) & set(p)):
        x, y = n[k], p[k]
        if x.shape != y.shape or x.dtype != y.dtype:
            difs.append((k, "forma/tipo", x.shape, y.shape, x.dtype, y.dtype))
        elif bool(mx.array_equal(x, y).item()):
            iguais += 1
        elif x.dtype in (mx.uint32, mx.int32, mx.uint8):
            difs.append((k, "elementos diferentes", int((x != y).sum().item()), x.size))
        else:
            difs.append((k, "max|dif|", float(mx.abs(x.astype(mx.float32) - y.astype(mx.float32)).max().item())))
    print(f"identicos: {iguais}/{len(set(n) & set(p))}; diferentes: {len(difs)}")
    for d in difs[:10]:
        print("  ", d)
    if a.nll:
        from avaliar import nll_por_janela
        from mlx_lm import load

        model, tok = load(a.novo)
        nll, _ = nll_por_janela(model, tok, 2048)
        ref = np.load(a.nll)
        print(f"NLL por janela: {len(nll)} janelas (padrao {len(ref)}); max|dif| = {np.abs(nll - ref).max():.3e}; "
              f"PPL novo {np.exp(nll.mean()):.4f} | padrao {np.exp(ref.mean()):.4f}")
        d = nll - ref                                   # diferenca pareada por janela; Q_novo/Q_padrao = exp(-media(d))
        idx = np.random.default_rng(0).integers(0, len(d), (10_000, len(d)))
        lo, hi = np.percentile(np.exp(-d[idx].mean(1)), [2.5, 97.5])
        print(f"razao Q_novo/Q_padrao = {np.exp(-d.mean()):.5f} (IC95% bootstrap pareado: {lo:.5f} a {hi:.5f}); "
              f"o IC {'contem' if lo <= 1 <= hi else 'NAO contem'} 1")


if __name__ == "__main__":
    main()
