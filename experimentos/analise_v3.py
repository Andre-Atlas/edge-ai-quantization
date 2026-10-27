"""Etapa 5 (extensao): aplica as regras de metodologia/protocolo-v3.md e gera resultados/hipoteses-v3.json.

  python experimentos/analise_v3.py

Nao altera a analise da v2 (analise.py, hipoteses.json), cujos vereditos permanecem. Diferencas da v3 (secao 3):
  - limiares da v1 em PROPORCAO dos modelos avaliaveis (80% corrobora, 60% contrarios refuta; N minimo 5);
  - cada modelo e classificado pelo IC95% (sucesso / contrario / indeterminado);
  - H2: celulas familia x metodo x bits, ordem por diferencas PAREADAS (mesmas janelas na familia);
  - H3: 1a metade na escala de NLL (S = (D2-D3)-(D3-D4)); 2a metade pela razao L_melhor/L_RTN a 3 bits;
  - IC dos metodos calibrados ampliado pela variancia entre sementes (sigma_cal, replicas de AWQ a 4 bits).
Bootstrap pareado sobre as janelas de 2048 tokens, 10 000 reamostragens; a MESMA matriz de indices e usada para todos os
modelos com o mesmo numero de janelas (pareamento dentro do modelo e dentro da familia, que usa o mesmo tokenizador).
"""
import csv
import json
from pathlib import Path

import numpy as np

from analise import ident

RAIZ = Path(__file__).resolve().parent
RES = RAIZ / "resultados"
CSV, NLL, SAIDA = RES / "resultados.csv", RES / "nll", RES / "hipoteses-v3.json"
B, LIMIAR, Z = 10_000, 0.95, 1.96
CORR, REF, NMIN = 0.80, 0.60, 5                   # limiares da v1 em proporcao (4/5 e 3/5); N minimo de avaliaveis
CIENTES = ["gptq", "gptq_fp32", "awq", "dwq"]     # H1: qualquer metodo ciente valido a 4 bits
APRENDEM = ["gptq_fp32", "dwq"]                   # H3, 2a metade (o GPTQ padrao nao existe a 3 bits)
CALIBRADOS = {"gptq", "gptq_fp32", "awq", "dwq"}
ANTIGOS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "HuggingFaceTB/SmolLM3-3B", "meta-llama/Llama-3.2-3B", "Qwen/Qwen3-4B"]
NOVOS = ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", "meta-llama/Llama-3.2-1B"]
MODELOS = ANTIGOS + NOVOS
FAMILIAS = {"Qwen3": ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"],
            "Qwen2.5": ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"]}
METODOS_H2 = ["rtn", "awq", "gptq_fp32"]


# ---------- regras puras (testadas em testes_v3.py) ----------

def ampliar(ic, extra_sd):
    """IC com a variancia da calibracao somada: meia-largura sqrt(h^2 + (Z*sd)^2), mesmo centro."""
    lo, hi = ic
    if extra_sd <= 0:
        return (lo, hi)
    c, h = (lo + hi) / 2, (hi - lo) / 2
    n = (h ** 2 + (Z * extra_sd) ** 2) ** 0.5
    return (c - n, c + n)


def classificar_h1(ic_rtn, ics_cientes, limiar=LIMIAR):
    if ic_rtn[0] >= limiar or (ics_cientes and all(ic[1] < limiar for ic in ics_cientes)):
        return "contrario"
    if ics_cientes and max(ics_cientes, key=lambda ic: ic[0])[0] >= limiar and ic_rtn[1] < limiar:
        return "sucesso"
    return "indeterminado"


def classificar_lado(ic, limite=0.0, maior_e_bom=True):
    """'sim' se o IC inteiro esta do lado esperado do limite, 'contrario' se inteiro do outro lado, senao 'indeterminado'."""
    lo, hi = ic
    if maior_e_bom:
        return "sim" if lo > limite else "contrario" if hi < limite else "indeterminado"
    return "sim" if hi < limite else "contrario" if lo > limite else "indeterminado"


def veredito_proporcao(n_sim, n_contra, n, corr=CORR, ref=REF, nmin=NMIN):
    if n < nmin:
        return "nao avaliavel (N insuficiente)"
    if n_contra / n >= ref:
        return "refutada"
    if n_sim / n >= corr:
        return "corroborada"
    return "inconclusiva"


def veredito_h2(n_ok, n_cel, nmin=6):
    if n_cel < nmin:
        return "nao avaliavel (celulas insuficientes)"
    f = n_ok / n_cel
    return "corroborada" if f >= 0.75 else "refutada" if f < 0.5 else "inconclusiva"


def veredito_h3(v1, v2):
    """Combina as duas metades: refutada se qualquer uma refuta; corroborada so se as duas corroboram."""
    if "refutada" in (v1, v2):
        return "refutada"
    if v1 == v2 == "corroborada":
        return "corroborada"
    if any(v.startswith("nao avaliavel") for v in (v1, v2)):
        return "nao avaliavel (N insuficiente)"
    return "inconclusiva"


# ---------- dados ----------

def carregar():
    validas = [r for r in csv.DictReader(open(CSV)) if r.get("ppl_wikitext2") and not r.get("status", "").startswith("falhou")]
    tab = {}
    for r in validas:
        p = NLL / f"{ident(r)}.npy"
        if p.exists():
            v = np.load(p).astype(np.float64)
            if np.isfinite(v).all():
                tab[(r["modelo"], r["metodo"], int(r["bits"]), r["seed"])] = v
    return tab


_IDX = {}


def indices(n):
    """Mesma matriz de reamostragem para todo vetor com n janelas (pareamento entre metodos e entre tamanhos da familia)."""
    if n not in _IDX:
        _IDX[n] = np.random.default_rng(0).integers(0, n, (B, n))
    return _IDX[n]


def q_boot(tab, m, met, bits, seed="0"):
    nf, nq = tab.get((m, "fp", 16, "0")), tab.get((m, met, bits, seed))
    if nf is None or nq is None or len(nf) != len(nq):
        return None, None
    idx = indices(len(nf))
    return float(np.exp(nf.mean() - nq.mean())), np.exp(nf[idx].mean(1) - nq[idx].mean(1))


def ic(x):
    return tuple(float(v) for v in np.percentile(x, [2.5, 97.5]))


def sigma_cal(tab):
    """Desvio-padrao de Q entre sementes (AWQ 4 bits, sementes 0-2), combinado entre os modelos com as tres sementes."""
    variancias, detalhe = [], {}
    for m in MODELOS:
        qs = [q_boot(tab, m, "awq", 4, s)[0] for s in ("0", "1", "2")]
        if all(q is not None for q in qs):
            variancias.append(np.var(qs, ddof=1))
            detalhe[m] = [round(q, 4) for q in qs]
    return (float(np.mean(variancias)) ** 0.5 if variancias else 0.0), detalhe


def ic_q(tab, m, met, bits, sc):
    q, x = q_boot(tab, m, met, bits)
    if q is None:
        return None, None
    i = ic(x)
    return q, (ampliar(i, sc) if met in CALIBRADOS else i)


# ---------- hipoteses ----------

def h1(tab, sc, modelos):
    ev, cont = {}, {"sucesso": 0, "contrario": 0, "indeterminado": 0}
    for m in modelos:
        q_rtn, ic_rtn = ic_q(tab, m, "rtn", 4, sc)
        cs = {met: ic_q(tab, m, met, 4, sc) for met in CIENTES}
        cs = {k: v for k, v in cs.items() if v[0] is not None}
        if q_rtn is None or not cs:
            ev[m] = "nao avaliavel"
            continue
        c = classificar_h1(ic_rtn, [v[1] for v in cs.values()])
        cont[c] += 1
        ev[m] = {"classe": c, "rtn": [round(q_rtn, 4), [round(v, 4) for v in ic_rtn]],
                 "cientes": {k: [round(v[0], 4), [round(x, 4) for x in v[1]]] for k, v in cs.items()}}
    n = sum(cont.values())
    return veredito_proporcao(cont["sucesso"], cont["contrario"], n), ev, f"{cont} de {n} avaliaveis"


def h2(tab, sc):
    ev, ok, n = {}, 0, 0
    for fam, (p, me, g) in FAMILIAS.items():
        for met in METODOS_H2:
            for bits in (4, 3):
                xs = [q_boot(tab, m, met, bits) for m in (p, me, g)]
                if any(x[0] is None for x in xs) or len({len(x[1]) for x in xs}) != 1:
                    continue
                extra = (2 ** 0.5) * sc if met in CALIBRADOS else 0.0
                d1, d2 = ampliar(ic(xs[1][1] - xs[0][1]), extra), ampliar(ic(xs[2][1] - xs[1][1]), extra)
                conf = d1[0] > 0 and d2[0] > 0
                n += 1
                ok += conf
                ev[f"{fam}-{met}-{bits}b"] = {"Q": [round(x[0], 4) for x in xs], "ic_medio_menos_pequeno": [round(v, 4) for v in d1],
                                              "ic_grande_menos_medio": [round(v, 4) for v in d2], "ordem_confirmada": bool(conf)}
    return veredito_h2(ok, n), ev, f"ordem confirmada em {ok} de {n} celulas validas"


def h3(tab, sc, modelos):
    ev, c1, c2 = {}, {"sim": 0, "contrario": 0, "indeterminado": 0}, {"sim": 0, "contrario": 0, "indeterminado": 0}
    for m in modelos:
        e = {}
        nf = tab.get((m, "fp", 16, "0"))
        rtn = {b: tab.get((m, "rtn", b, "0")) for b in (4, 3, 2)}
        if nf is not None and all(v is not None and len(v) == len(nf) for v in rtn.values()):
            idx = indices(len(nf))
            d = {b: rtn[b][idx].mean(1) - nf[idx].mean(1) for b in rtn}
            s_ic = ic((d[2] - d[3]) - (d[3] - d[4]))
            k = classificar_lado(s_ic, 0.0, maior_e_bom=True)
            c1[k] += 1
            e["S_ic"], e["superlinear"] = [round(v, 4) for v in s_ic], k
        q3, x3 = q_boot(tab, m, "rtn", 3)
        cand = [(met, q_boot(tab, m, met, 3)) for met in APRENDEM]
        cand = [(met, q) for met, q in cand if q[0] is not None]
        if q3 is not None and cand and q3 < 1:
            met, (qb, xb) = max(cand, key=lambda c: c[1][0])
            razao = (1 - xb) / (1 - x3)
            r_ic = ampliar(ic(razao), sc / (1 - q3))   # sd de Q propagado para a razao: sd/L_RTN
            k = classificar_lado(r_ic, 0.8, maior_e_bom=False)
            c2[k] += 1
            e["L3_razao_melhor_rtn"], e["melhor"], e["reduz_20pct"] = [round(v, 4) for v in r_ic], met, k
        ev[m] = e or "nao avaliavel"
    n1, n2 = sum(c1.values()), sum(c2.values())
    v1 = veredito_proporcao(c1["sim"], c1["contrario"], n1)
    v2 = veredito_proporcao(c2["sim"], c2["contrario"], n2)
    return veredito_h3(v1, v2), ev, f"1a metade {v1} {c1} (N={n1}); 2a metade {v2} {c2} (N={n2})"


def main():
    if not CSV.exists():
        raise SystemExit("resultados.csv ainda nao existe")
    tab = carregar()
    sc, sd = sigma_cal(tab)
    print(f"sigma_cal (AWQ 4 bits, sementes 0-2) = {sc:.4f} | modelos: {sd}")
    saida = {"protocolo": "protocolo-v3", "sigma_cal": sc, "sigma_cal_detalhe": sd, "hipoteses": {}, "robustez_so_novos": {}}
    for nome, f in [("H1", lambda: h1(tab, sc, MODELOS)), ("H2", lambda: h2(tab, sc)), ("H3", lambda: h3(tab, sc, MODELOS))]:
        v, ev, det = f()
        saida["hipoteses"][nome] = {"veredito": v, "resumo": det, "evidencia": ev}
        print(f"{nome}: {v} | {det}")
    for nome, f in [("H1", lambda: h1(tab, sc, NOVOS)), ("H3", lambda: h3(tab, sc, NOVOS))]:
        v, ev, det = f()                                 # N = 4 < 5: so a classificacao por modelo, sem veredito (secao 3)
        saida["robustez_so_novos"][nome] = {"resumo": det, "evidencia": ev}
        print(f"{nome} (robustez, so os 4 novos; sem veredito): {det}")
    SAIDA.write_text(json.dumps(saida, ensure_ascii=False, indent=1, default=float))
    print(f"-> {SAIDA.relative_to(RAIZ.parent)}")


if __name__ == "__main__":
    main()
