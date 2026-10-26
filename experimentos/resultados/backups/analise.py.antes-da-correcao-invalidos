"""Etapa 5: aplica os criterios de metodologia/protocolo-v2.md (limiares da protocolo-v1) e gera resultados/hipoteses.json.

  python experimentos/analise.py

Q = PPL_fp / PPL_q com IC de 95% por bootstrap PAREADO sobre as janelas de 2048 tokens
(as mesmas janelas sao reamostradas no fp e no quantizado; 10 000 reamostragens).
Veredito so "corroborada"/"refutada" quando os ICs ficam inteiros de um lado do limiar.

Regras da v2 (secao 5): os limiares absolutos da v1 sao mantidos; modelos sem os dados necessarios sao
"nao avaliaveis" e nao contam como sucesso. Um dado pode estar (a) presente e valido, (b) PENDENTE (ausente, mas
a combinacao nao esta em `excluir` de matriz.json: ainda pode chegar) ou (c) permanentemente ausente (em `excluir`).
Contagens monotonas decidem cedo: refutada/corroborada assim que a contagem basta; "aguardando dados" so quando
ainda ha dado pendente que poderia mudar o veredito. Linhas com status "falhou*" ficam de fora.
"""
import csv
import json
from pathlib import Path

import numpy as np

from matriz import excluido

RAIZ = Path(__file__).resolve().parent
RES = RAIZ / "resultados"
CSV, NLL, SAIDA = RES / "resultados.csv", RES / "nll", RES / "hipoteses.json"
B, LIMIAR = 10_000, 0.95
CIENTES = ["gptq", "awq", "dwq"]
MODELOS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "HuggingFaceTB/SmolLM3-3B", "meta-llama/Llama-3.2-3B", "Qwen/Qwen3-4B"]
SEPARADOS = ["HuggingFaceTB/SmolLM3-3B"]   # caso a parte (v2): fora das contagens de H1 e H3; entra so na sensibilidade
CONTADOS = [m for m in MODELOS if m not in SEPARADOS]
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


CFG = json.load(open(RAIZ / "matriz.json"))


def pendente(L, m, metodo, bits):
    """Dado ainda possivel: ausente do CSV valido e a combinacao nao esta em `excluir` (protocolo-v2, secao 3)."""
    return pega(L, m, metodo, bits) is None and not excluido(CFG, dict(modelo=m, metodo=metodo, bits=bits))


def h1(L, modelos=CONTADOS):
    """Criterio v1 com limiares absolutos; 'algum metodo ciente' = os disponiveis em 4 bits (v2, secao 5)."""
    ev, sucesso, rtn_ok, nenhum, pend, nao_aval = {}, 0, 0, 0, [], []
    for m in modelos:
        rtn = pega(L, m, "rtn", 4)
        cs = [r for r in (pega(L, m, x, 4) for x in CIENTES) if r]
        faltam = [x for x in CIENTES if pendente(L, m, x, 4)]
        if not rtn:
            pend.append(m)
            continue
        rtn_ok += acima(rtn)                           # definitivo: independe dos metodos cientes de dados
        if not cs:
            if not acima(rtn):
                (pend if faltam else nao_aval).append(m)
            continue
        melhor = max(cs, key=lambda r: r["Q"])
        exito = acima(melhor) and abaixo(rtn)
        ev[m] = {"rtn": resumo(rtn), "melhor_ciente": resumo(melhor), "cientes_pendentes": faltam}
        sucesso += exito
        if not faltam:
            nenhum += all(abaixo(r) for r in cs)       # 'nenhum' so e definitivo com todos os metodos possiveis presentes
        if not (exito or acima(rtn) or not faltam):
            pend.append(m)
    det = (f"criterio atendido em {sucesso}; RTN >= 0,95 em {rtn_ok}; nenhum ciente >= 0,95 em {nenhum}; "
           f"nao avaliaveis: {nao_aval}; pendentes: {pend}")
    if rtn_ok >= 3 or nenhum >= 3:
        return "refutada", ev, det
    if sucesso >= 4:
        return "corroborada", ev, det
    return ("aguardando dados" if pend else "inconclusiva"), ev, det


def h2(L):
    """Celulas = RTN x {4, 3 bits} nos tres Qwen3; GPTQ/AWQ/DWQ so em 0,6B e 1,7B: suplementar, fora da fracao (v2, secao 5)."""
    celulas, ok, ev = 0, 0, {}
    for bits in (4, 3):
        rs = [pega(L, m, "rtn", bits) for m in QWEN]
        if not all(rs) or any(r["ic"][0] is None for r in rs):
            return "aguardando dados", ev, f"falta rtn {bits}b em algum Qwen3"
        celulas += 1
        ordem = all(a["ic"][1] < b["ic"][0] for a, b in zip(rs, rs[1:]))
        ok += ordem
        ev[f"rtn-{bits}b"] = {"Q": [round(r["Q"], 4) for r in rs], "ordem_confirmada": bool(ordem)}
    for met in CIENTES:
        for bits in (4, 3):
            a, b = (pega(L, m, met, bits) for m in QWEN[:2])
            if a and b and a["ic"][0] is not None and b["ic"][0] is not None:
                ev.setdefault("suplementar_0.6B_menor_1.7B", {})[f"{met}-{bits}b"] = bool(a["ic"][1] < b["ic"][0])
    frac = ok / celulas
    v = "corroborada" if frac >= 0.75 else "refutada" if frac < 0.5 else "inconclusiva"
    return v, ev, f"ordem 0,6B < 1,7B < 4B confirmada em {ok}/{celulas} celulas do RTN ({frac:.0%}); GPTQ/AWQ/DWQ so como suplementar"


def h3(L, modelos=CONTADOS):
    """1a metade: criterio original (L = 1 - Q) sobre os modelos com RTN completo. 2a metade: melhor entre GPTQ e DWQ
    disponiveis em 3 bits contra o RTN; contagens absolutas da v1 (v2, secao 5). Ver tambem analise_h3_nll.py."""
    Ls = {b: [] for b in (4, 3, 2)}
    reduz, nao_reduz, ev, pend, nao_aval = 0, 0, {}, [], []
    pend_rtn = False
    for m in modelos:
        rtn = {b: pega(L, m, "rtn", b) for b in (4, 3, 2)}
        if all(rtn.values()):
            for b in Ls:
                Ls[b].append(1 - rtn[b]["Q"])
        else:
            pend_rtn = True
        if not rtn[3]:
            pend.append(m)
            continue
        cand = [r for r in (pega(L, m, x, 3) for x in ("gptq", "dwq")) if r]
        faltam = [x for x in ("gptq", "dwq") if pendente(L, m, x, 3)]
        if not cand:
            (pend if faltam else nao_aval).append(m)
            continue
        l_rtn, l_best = 1 - rtn[3]["Q"], min(1 - r["Q"] for r in cand)
        red = l_best <= 0.8 * l_rtn
        reduz += red
        if not red and not faltam:
            nao_reduz += 1
        elif not red:
            pend.append(m)
        ev[m] = {"L3_rtn": round(l_rtn, 4), "L3_melhor_gptq_dwq": round(l_best, 4), "reduz_20pct": bool(red), "pendentes": faltam}
    med = {b: float(np.mean(v)) for b, v in Ls.items() if v}
    superlinear = bool(med) and (med[2] - med[3]) > (med[3] - med[4])
    inc = f"incrementos medios de L (RTN, {len(Ls[4])} modelos): 4->3 {med[3]-med[4]:.4f}, 3->2 {med[2]-med[3]:.4f}" if med else "sem RTN completo"
    det = f"{inc}; reducao >=20% em 3b em {reduz}, sem reducao em {nao_reduz}; nao avaliaveis: {nao_aval}; pendentes: {pend}"
    if nao_reduz >= 3 or (med and not pend_rtn and not superlinear):
        return "refutada", ev, det
    if pend_rtn or pend or not med:
        return "aguardando dados", ev, det
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
    saida = {"protocolo": "protocolo-v2","n_configuracoes": len(L), "hipoteses": {}}
    for nome, f in [("H1", h1), ("H2", h2), ("H3", h3), ("H4", h4)]:
        v, ev, det = f(L)
        saida["hipoteses"][nome] = {"veredito": v, "resumo": det, "evidencia": ev}
        print(f"{nome}: {v} | {det}")
    saida["sensibilidade_com_caso_separado"] = {}
    for nome, f in [("H1", h1), ("H3", h3)]:           # o veredito mudaria se o SmolLM3 fosse contado? (gatilho da analise B)
        v, ev, det = f(L, MODELOS)
        saida["sensibilidade_com_caso_separado"][nome] = {"veredito": v, "resumo": det}
        print(f"{nome} (sensibilidade, com {', '.join(m.split('/')[-1] for m in SEPARADOS)}): {v} | {det}")
    saida["pareto"] = pareto(L)
    SAIDA.write_text(json.dumps(saida, ensure_ascii=False, indent=1, default=float))
    print(f"-> {SAIDA.relative_to(RAIZ.parent)}")


if __name__ == "__main__":
    main()
