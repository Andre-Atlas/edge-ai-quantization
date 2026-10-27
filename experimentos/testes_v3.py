"""Testes sinteticos das regras da v3 (analise_v3.py) e do empacotamento de 3 bits (gptq_camada.empacotar). Sem modelos: ~10 s.

  python experimentos/testes_v3.py
Os dados de NLL sao gerados aqui (nada e lido de resultados/); cada caso fixa o veredito esperado pela secao 3 do
protocolo-v3, inclusive os ramos de N insuficiente e de IC que cruza o limiar.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analise_v3 as A  # noqa: E402

falhas = []


def checar(nome, ok, detalhe=""):
    print(("PASSOU " if ok else "FALHOU ") + nome + (": " + detalhe if detalhe else ""), flush=True)
    if not ok:
        falhas.append(nome)


# ---------- regras puras ----------
checar("ampliar sem sigma nao muda", A.ampliar((0.9, 1.0), 0) == (0.9, 1.0))
lo, hi = A.ampliar((0.94, 0.96), 0.01)
checar("ampliar mantem centro e alarga", abs((lo + hi) / 2 - 0.95) < 1e-12 and hi - lo > 0.02, f"{lo:.4f}-{hi:.4f}")
checar("H1 sucesso", A.classificar_h1((0.90, 0.92), [(0.951, 0.96), (0.80, 0.85)]) == "sucesso")
checar("H1 contrario por RTN >= 0,95", A.classificar_h1((0.951, 0.97), [(0.96, 0.98)]) == "contrario")
checar("H1 contrario por nenhum ciente", A.classificar_h1((0.80, 0.82), [(0.90, 0.949), (0.93, 0.94)]) == "contrario")
checar("H1 indeterminado (IC cruza 0,95)", A.classificar_h1((0.80, 0.82), [(0.945, 0.955)]) == "indeterminado")
checar("lado: sim/contrario/indet", [A.classificar_lado(i) for i in [(0.1, 0.2), (-0.2, -0.1), (-0.1, 0.1)]]
       == ["sim", "contrario", "indeterminado"])
checar("lado invertido (razao < 0,8)", A.classificar_lado((0.5, 0.7), 0.8, maior_e_bom=False) == "sim"
       and A.classificar_lado((0.85, 0.9), 0.8, maior_e_bom=False) == "contrario")
checar("proporcao corrobora 8/9", A.veredito_proporcao(8, 0, 9) == "corroborada")
checar("proporcao 7/9 nao corrobora (77,8% < 80%)", A.veredito_proporcao(7, 1, 9) == "inconclusiva")
checar("proporcao refuta 6/9 contrarios", A.veredito_proporcao(2, 6, 9) == "refutada")
checar("proporcao N<5", A.veredito_proporcao(4, 0, 4).startswith("nao avaliavel"))
checar("H2 limiares", [A.veredito_h2(k, 8) for k in (6, 5, 3)] == ["corroborada", "inconclusiva", "refutada"]
       and A.veredito_h2(5, 5).startswith("nao avaliavel"))
checar("H3 combina metades", A.veredito_h3("corroborada", "corroborada") == "corroborada"
       and A.veredito_h3("corroborada", "refutada") == "refutada" and A.veredito_h3("corroborada", "inconclusiva") == "inconclusiva")


# ---------- ponta a ponta com NLL sintetica ----------
rng = np.random.default_rng(1)
NJ = 140


def janelas(base, ruido=0.02):
    return base + rng.normal(0, ruido, NJ)


def tab_sintetica(perda4_rtn, perda4_ciente, perda3_rtn, perda3_ciente, perda2_rtn, modelos):
    """perdas em nats/token (D_b); Q = exp(-D). Mesmos valores em todos os modelos, salvo o acrescimo por tamanho."""
    t = {}
    pos = {m: i for fam in A.FAMILIAS.values() for i, m in enumerate(fam)}
    for m in modelos:
        f = janelas(3.0)
        t[(m, "fp", 16, "0")] = f
        esc = 1 - 0.15 * pos.get(m, 0)                 # na familia, o menor perde mais (ordem da H2 verdadeira)
        t[(m, "rtn", 4, "0")] = f + perda4_rtn * esc + rng.normal(0, 0.002, NJ)
        t[(m, "rtn", 3, "0")] = f + perda3_rtn * esc + rng.normal(0, 0.002, NJ)
        t[(m, "rtn", 2, "0")] = f + perda2_rtn * esc + rng.normal(0, 0.002, NJ)
        for met in ("awq", "gptq_fp32"):
            t[(m, met, 4, "0")] = f + perda4_ciente * esc + rng.normal(0, 0.002, NJ)
            t[(m, met, 3, "0")] = f + perda3_ciente * esc + rng.normal(0, 0.002, NJ)
    return t


# cenario 1: H1 verdadeira (RTN perde ~10%, ciente ~2%), H3 verdadeira (superlinear; ciente reduz L a 3 bits), H2 verdadeira
t1 = tab_sintetica(0.11, 0.02, 0.4, 0.15, 2.0, A.MODELOS)
v, _, det = A.h1(t1, 0.0, A.MODELOS)
checar("ponta a ponta H1 corroborada", v == "corroborada", det)
v, _, det = A.h2(t1, 0.0)
checar("ponta a ponta H2 corroborada", v == "corroborada", det)
v, _, det = A.h3(t1, 0.0, A.MODELOS)
checar("ponta a ponta H3 corroborada", v == "corroborada", det)

# cenario 2: RTN ja retem >= 95% (perda 1%) -> H1 refutada; incrementos lineares (0,1 / 0,2 / 0,3) -> H3 refutada
t2 = tab_sintetica(0.01, 0.005, 0.2, 0.19, 0.3, A.MODELOS)
v, _, det = A.h1(t2, 0.0, A.MODELOS)
checar("ponta a ponta H1 refutada", v == "refutada", det)
v, _, det = A.h3(t2, 0.0, A.MODELOS)
checar("ponta a ponta H3 refutada", v == "refutada", det)

# cenario 3: so os 4 novos -> N < 5 em H1 (sem veredito); sigma_cal grande deixa a H1 indeterminada
v, _, det = A.h1(t1, 0.0, A.NOVOS)
checar("so novos: N insuficiente", v.startswith("nao avaliavel"), det)
v, _, det = A.h1(tab_sintetica(0.11, 0.045, 0.4, 0.15, 2.0, A.MODELOS), 0.05, A.MODELOS)
checar("sigma_cal grande -> nao corrobora", v != "corroborada", det)

# empacotamento de 3 bits = formato do mx.quantize
import mlx.core as mx  # noqa: E402
from gptq_camada import empacotar  # noqa: E402
from mlx_lm.quant.gptq import quantize as q_mlxlm  # noqa: E402

mx.random.seed(2)
# (a) FORMATO, exato: inteiros com 0 e n_bins em todo grupo (escala 1, vies 0 sem arredondamento) -> mesmos bits do mx.quantize
ok = True
for forma in [(64, 128), (300, 2560)]:
    for bits in (2, 3, 4, 6, 8):
        n = 2**bits - 1
        qi = np.random.default_rng(bits).integers(0, n + 1, forma)
        qi[:, ::64], qi[:, 1::64] = 0, n
        w = mx.array(qi.astype(np.float32))
        wq, s, b = mx.quantize(w, bits=bits, group_size=64)
        ok &= bool(mx.array_equal(empacotar(w, bits, s, b), wq))
checar("empacotar: formato identico ao mx.quantize (2,3,4,6,8 bits)", ok)
# (b) valores reais: identico ao quantize do mlx-lm (2,4,8) e ao mx.quantize salvo empates de arredondamento (<0,01% das palavras)
ok, pior = True, 0.0
for forma in [(64, 128), (300, 2560)]:
    w = mx.random.normal(forma)
    for bits in (2, 3, 4, 6, 8):
        wq, s, b = mx.quantize(w, bits=bits, group_size=64)
        p = empacotar(w, bits, s, b)
        if bits in (2, 4, 8):
            ok &= bool(mx.array_equal(p, q_mlxlm(w, bits, s, b)))
        pior = max(pior, float((np.array(p) != np.array(wq)).mean()))
checar("empacotar: igual ao mlx-lm (2,4,8) e empates < 0,01% (3,6)", ok and pior < 1e-4, f"pior fracao diferente {pior:.6f}")

print(f"\n{len(falhas)} falha(s)" + (f": {falhas}" if falhas else ""))
sys.exit(1 if falhas else 0)
