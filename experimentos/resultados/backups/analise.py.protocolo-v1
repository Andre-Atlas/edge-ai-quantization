"""Etapa 5: aplica os criterios de metodologia/protocolo.md (protocolo-v1) e gera resultados/hipoteses.json.

  python experimentos/analise.py

Q = PPL_fp / PPL_q com IC de 95% por bootstrap PAREADO sobre as janelas de 2048 tokens
(as mesmas janelas sao reamostradas no fp e no quantizado; 10 000 reamostragens).
Veredito so "corroborada"/"refutada" quando os ICs ficam inteiros de um lado do limiar;
com dados faltando, "aguardando dados".
"""
import csv
import json
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent
RES = RAIZ / "resultados"
CSV, NLL, SAIDA = RES / "resultados.csv", RES / "nll", RES / "hipoteses.json"
B, LIMIAR = 10_000, 0.95
CIENTES = ["gptq", "awq", "dwq"]
MODELOS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "HuggingFaceTB/SmolLM3-3B", "meta-llama/Llama-3.2-3B", "Qwen/Qwen3-4B"]
QWEN = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"]
GRANDES = ["HuggingFaceTB/SmolLM3-3B", "meta-llama/Llama-3.2-3B", "Qwen/Qwen3-4B"]
rng = np.random.default_rng(0)


def ident(r):
    base = r["modelo"].split("/")[-1] + "-" + r["metodo"]
    return base if r["metodo"] == "fp" else f"{base}-{r['bits']}b-g{r['grupo']}-s{r['seed']}"


def carregar():
    linhas = [r for r in csv.DictReader(open(CSV)) if r.get("ppl_wikitext2") and not r.get("status", "").startswith("falhou")]
    fp = {r["modelo"]: r for r in linhas if r["metodo"] == "fp"}
    for r in linhas:
        r["bits"] = int(r["bits"])
        f = fp.get(r["modelo"])
        r["Q"] = float(f["ppl_wikitext2"]) / float(r["ppl_wikitext2"]) if f else None
        r["ic"] = (None, None)
        a, b = NLL / f"{ident(f)}.npy" if f else None, NLL / f"{ident(r)}.npy"
        if f and a.exists() and b.exists():
            nf, nq = np.load(a), np.load(b)
            if len(nf) == len(nq):
                idx = rng.integers(0, len(nf), (B, len(nf)))
                qs = np.exp(nf[idx].mean(1) - nq[idx].mean(1))  # PPL_fp/PPL_q = exp(nll_fp - nll_q)
                r["ic"] = tuple(np.percentile(qs, [2.5, 97.5]).round(5))
    return linhas


def pega(linhas, modelo, metodo, bits, seed="0"):
    for r in linhas:
        if r["modelo"] == modelo and r["metodo"] == metodo and r["bits"] == bits and r["seed"] == seed:
            return r


acima = lambda r: r["ic"][0] is not None and r["ic"][0] >= LIMIAR
abaixo = lambda r: r["ic"][1] is not None and r["ic"][1] < LIMIAR


def resumo(r):
    return None if r is None else {"metodo": r["metodo"], "bits": r["bits"], "Q": round(r["Q"], 4), "ic95": r["ic"]}


def h1(L):
    ev, sucesso, rtn_ok, nenhum = {}, 0, 0, 0
    faltam = []
    for m in MODELOS:
        rtn = pega(L, m, "rtn", 4)
        cs = [r for r in (pega(L, m, x, 4) for x in CIENTES) if r]
        if not rtn or len(cs) < len(CIENTES):
            faltam.append(m)
            continue
        melhor = max(cs, key=lambda r: r["Q"])
        ev[m] = {"rtn": resumo(rtn), "melhor_ciente": resumo(melhor)}
        sucesso += acima(melhor) and abaixo(rtn)
        rtn_ok += acima(rtn)
        nenhum += all(abaixo(r) for r in cs)
    if faltam:
        return "aguardando dados", ev, f"faltam modelos: {faltam}"
    if rtn_ok >= 3 or nenhum >= 3:
        return "refutada", ev, f"RTN >= 0,95 em {rtn_ok} modelos; nenhum ciente >= 0,95 em {nenhum}"
    if sucesso >= 4:
        return "corroborada", ev, f"criterio atendido em {sucesso}/5 modelos"
    return "inconclusiva", ev, f"criterio atendido em {sucesso}/5 modelos"


def h2(L):
    celulas, ok, ev = 0, 0, {}
    for met in ["rtn", *CIENTES]:
        for bits in (4, 3):
            rs = [pega(L, m, met, bits) for m in QWEN]
            if not all(rs) or any(r["ic"][0] is None for r in rs):
                return "aguardando dados", ev, f"falta {met} {bits}b em algum Qwen3"
            celulas += 1
            ordem = all(a["ic"][1] < b["ic"][0] for a, b in zip(rs, rs[1:]))
            ok += ordem
            ev[f"{met}-{bits}b"] = {"Q": [round(r["Q"], 4) for r in rs], "ordem_confirmada": bool(ordem)}
    frac = ok / celulas
    v = "corroborada" if frac >= 0.75 else "refutada" if frac < 0.5 else "inconclusiva"
    return v, ev, f"ordem 0,6B < 1,7B < 4B confirmada em {ok}/{celulas} celulas ({frac:.0%})"


def h3(L):
    Ls = {b: [] for b in (4, 3, 2)}
    reduz, ev = 0, {}
    for m in MODELOS:
        rtn = {b: pega(L, m, "rtn", b) for b in (4, 3, 2)}
        g3, d3 = pega(L, m, "gptq", 3), pega(L, m, "dwq", 3)
        if not all(rtn.values()) or not (g3 and d3):
            return "aguardando dados", ev, f"faltam dados de {m}"
        for b in Ls:
            Ls[b].append(1 - rtn[b]["Q"])
        l_rtn, l_best = 1 - rtn[3]["Q"], min(1 - g3["Q"], 1 - d3["Q"])
        reduz += l_best <= 0.8 * l_rtn
        ev[m] = {"L_rtn": {b: round(1 - rtn[b]["Q"], 4) for b in (4, 3, 2)}, "L3_melhor_gptq_dwq": round(l_best, 4)}
    med = {b: float(np.mean(v)) for b, v in Ls.items()}
    superlinear = (med[2] - med[3]) > (med[3] - med[4])
    det = f"incrementos medios de L (RTN): 4->3 {med[3]-med[4]:.4f}, 3->2 {med[2]-med[3]:.4f}; reducao >=20% em 3b em {reduz}/5 modelos"
    if not superlinear or reduz < 3:
        return "refutada", ev, det
    return ("corroborada" if reduz >= 4 else "inconclusiva"), ev, det


def h4(L):
    ev, Es = {}, []
    for m in GRANDES:
        fp = pega(L, m, "fp", 16)
        qs = [r for r in (pega(L, m, x, 4) for x in ["rtn", *CIENTES]) if r and r.get("decode_tps")]
        if not fp or not fp.get("decode_tps") or not qs:
            return "aguardando dados", ev, f"falta velocidade de {m}"
        e = [(float(q["decode_tps"]) / float(fp["decode_tps"])) / (float(fp["tamanho_gb"]) / float(q["tamanho_gb"])) for q in qs]
        ev[m] = {"E_por_metodo": {q["metodo"]: round(x, 3) for q, x in zip(qs, e)}, "E_media": round(float(np.mean(e)), 3)}
        Es.append(np.mean(e))
    det = "E media por modelo: " + ", ".join(f"{x:.2f}" for x in Es) + " (interpretacao: limiar vale para os 3 modelos)"
    if all(x >= 0.6 for x in Es):
        return "corroborada", ev, det
    if all(x < 0.4 for x in Es):
        return "refutada", ev, det
    return "inconclusiva", ev, det


def pareto(L):
    rs = [r for r in L if r["Q"] is not None and r.get("tamanho_gb")]
    front = []
    for m in {r["modelo"] for r in rs}:
        g = [r for r in rs if r["modelo"] == m]
        for a in g:
            if not any(float(b["tamanho_gb"]) <= float(a["tamanho_gb"]) and b["Q"] >= a["Q"] and b is not a and
                       (float(b["tamanho_gb"]) < float(a["tamanho_gb"]) or b["Q"] > a["Q"]) for b in g):
                front.append({"modelo": m, "metodo": a["metodo"], "bits": a["bits"], "tamanho_gb": float(a["tamanho_gb"]), "Q": round(a["Q"], 4)})
    return sorted(front, key=lambda x: (x["modelo"], x["tamanho_gb"]))


def main():
    if not CSV.exists():
        raise SystemExit("resultados.csv ainda nao existe (rode a matriz)")
    L = carregar()
    saida = {"protocolo": "protocolo-v1", "n_configuracoes": len(L), "hipoteses": {}}
    for nome, f in [("H1", h1), ("H2", h2), ("H3", h3), ("H4", h4)]:
        v, ev, det = f(L)
        saida["hipoteses"][nome] = {"veredito": v, "resumo": det, "evidencia": ev}
        print(f"{nome}: {v} | {det}")
    saida["pareto"] = pareto(L)
    SAIDA.write_text(json.dumps(saida, ensure_ascii=False, indent=1, default=float))
    print(f"-> {SAIDA.relative_to(RAIZ.parent)}")


if __name__ == "__main__":
    main()
