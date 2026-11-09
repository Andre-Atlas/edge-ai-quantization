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
from typing import Any, Sequence

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


INVALIDOS = set()   # (modelo, metodo, bits, seed) de jobs JA EXECUTADOS e invalidos (status falhou:invalido-*): nao sao "pendentes"


def carregar() -> list[dict[str, Any]]:
    with open(CSV, encoding="utf-8") as f_csv:
        todas = list(csv.DictReader(f_csv))
    INVALIDOS.clear()
    INVALIDOS.update((r["modelo"], r["metodo"], int(r["bits"]), r["seed"]) for r in todas
                     if r.get("status", "").startswith("falhou:invalido"))
    metodos_v2 = {"fp", "rtn", *CIENTES}
    linhas = [r for r in todas if r.get("ppl_wikitext2") and not r.get("status", "").startswith("falhou")
              and r.get("modelo") in MODELOS and r.get("metodo") in metodos_v2]
    fp = {r["modelo"]: r for r in linhas if r["metodo"] == "fp"}
    for r in linhas:
        r["bits"] = int(r["bits"])
        f = fp.get(r["modelo"])
        r["Q"] = float(f["ppl_wikitext2"]) / float(r["ppl_wikitext2"]) if f else None
        r["Q_acc"] = (float(r["acc_media"]) / float(f["acc_media"])
                      if f and r.get("acc_media") and f.get("acc_media") and float(f["acc_media"]) > 0 else None)
        r["ic"] = (None, None)
        a, b = NLL / f"{ident(f)}.npy" if f else None, NLL / f"{ident(r)}.npy"
        if f and a.exists() and b.exists():
            nf, nq = np.load(a), np.load(b)
            if len(nf) == len(nq):
                idx = rng.integers(0, len(nf), (B, len(nf)))
                qs = np.exp(nf[idx].mean(1) - nq[idx].mean(1))  # PPL_fp/PPL_q = exp(nll_fp - nll_q)
                r["ic"] = tuple(np.percentile(qs, [2.5, 97.5]).round(5))
    return linhas


def pega(linhas: Sequence[dict[str, Any]], modelo: str, metodo: str, bits: int, seed: str = "0") -> dict[str, Any] | None:
    for r in linhas:
        if r["modelo"] == modelo and r["metodo"] == metodo and r["bits"] == bits and r["seed"] == seed:
            return r
    return None


def acima(r: dict[str, Any]) -> bool:
    return r["ic"][0] is not None and r["ic"][0] >= LIMIAR


def abaixo(r: dict[str, Any]) -> bool:
    return r["ic"][1] is not None and r["ic"][1] < LIMIAR


def resumo(r: dict[str, Any] | None) -> dict[str, Any] | None:
    return None if r is None else {"metodo": r["metodo"], "bits": r["bits"], "Q": round(r["Q"], 4), "ic95": r["ic"]}


with open(RAIZ / "matriz.json", encoding="utf-8") as _f_cfg:
    CFG = json.load(_f_cfg)


def pendente(L, m, metodo, bits):
    """Dado ainda possivel: ausente do CSV valido, fora de `excluir` (protocolo-v2, secao 3) e NAO executado-e-invalido
    (um job que ja rodou e saiu com NaN e um resultado definitivo de falha, nao um dado que ainda pode chegar)."""
    return (pega(L, m, metodo, bits) is None and (m, metodo, bits, "0") not in INVALIDOS
            and not excluido(CFG, dict(modelo=m, metodo=metodo, bits=bits)))


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


def _ranks(xs: Sequence[float] | np.ndarray) -> np.ndarray:
    """Postos medios (1..n) com tratamento de empates para correlacao de Spearman."""
    arr = np.asarray(xs, dtype=np.float64)
    ordem = np.argsort(arr)
    ranks = np.empty_like(arr)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[ordem[j + 1]] == arr[ordem[i]]:
            j += 1
        ranks[ordem[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _calcular_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    """Correlacao linear de Pearson protegida contra variancia nula ou N < 2."""
    if len(x) < 2 or len(y) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    r = float(np.corrcoef(x, y)[0, 1])
    return None if np.isnan(r) else round(r, 4)


def concordancia_acuracia(L: Sequence[dict[str, Any]], modelos: Sequence[str] | None = None) -> dict[str, Any]:
    """Checagem diagnostica secundaria (protocolo-v2, Etapa 5): concordancia entre Q = PPL_fp/PPL_q e Q_acc = Acc_q/Acc_fp."""
    if modelos is None:
        modelos = MODELOS
    pares = [r for r in L if r.get("metodo") in ("rtn", *CIENTES) and r.get("seed") == "0"
             and r.get("Q") is not None and r.get("Q_acc") is not None and r.get("modelo") in modelos]
    qs = np.array([float(r["Q"]) for r in pares], dtype=np.float64)
    qas = np.array([float(r["Q_acc"]) for r in pares], dtype=np.float64)
    pearson = _calcular_corr(qs, qas)
    spearman = _calcular_corr(_ranks(qs), _ranks(qas))
    por_modelo, conc_limiar, total_4b, conc_melhor, total_melhor = {}, 0, 0, 0, 0
    for m in modelos:
        m4 = [r for r in pares if r["modelo"] == m and int(r["bits"]) == 4]
        if not m4:
            continue
        met_det = {}
        for r in m4:
            c_lim = (float(r["Q"]) >= LIMIAR) == (float(r["Q_acc"]) >= LIMIAR)
            conc_limiar += int(c_lim)
            total_4b += 1
            met_det[r["metodo"]] = {
                "Q": round(float(r["Q"]), 4),
                "Q_acc": round(float(r["Q_acc"]), 4),
                "concorda_limiar_95pct": bool(c_lim),
            }
        cientes_m = [r for r in m4 if r["metodo"] in CIENTES]
        melhor_q = max(cientes_m, key=lambda r: float(r["Q"]))["metodo"] if cientes_m else None
        melhor_qa = max(cientes_m, key=lambda r: float(r["Q_acc"]))["metodo"] if cientes_m else None
        coincide = (melhor_q == melhor_qa) if melhor_q else None
        if coincide is not None:
            conc_melhor += int(coincide)
            total_melhor += 1
        por_modelo[m] = {
            "metodos_4b": met_det,
            "melhor_ciente_Q": melhor_q,
            "melhor_ciente_Q_acc": melhor_qa,
            "melhor_ciente_coincide": coincide,
        }
    return {
        "n_pares": len(pares),
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "concordancia_limiar_4b": f"{conc_limiar}/{total_4b}",
        "concordancia_melhor_ciente_4b": f"{conc_melhor}/{total_melhor}",
        "por_modelo_4b": por_modelo,
    }


def variancia_replicas_awq(L: Sequence[dict[str, Any]], modelos: Sequence[str] | None = None) -> dict[str, Any]:
    """Variancia entre sementes de calibracao (AWQ 4 bits, sementes 0, 1 e 2) em PPL, Q e Q_acc."""
    if modelos is None:
        modelos = MODELOS
    por_modelo: dict[str, Any] = {}
    vars_q: list[float] = []
    for m in modelos:
        rs = [pega(L, m, "awq", 4, s) for s in ("0", "1", "2")]
        if not all(rs):
            continue
        ppls = [float(r["ppl_wikitext2"]) for r in rs if r is not None]
        qs = [float(r["Q"]) for r in rs if r is not None and r.get("Q") is not None]
        qas = [float(r["Q_acc"]) for r in rs if r is not None and r.get("Q_acc") is not None]
        media_ppl = float(np.mean(ppls))
        dp_ppl = float(np.std(ppls, ddof=1))
        if len(qs) >= 2:
            var_q = float(np.var(qs, ddof=1))
            vars_q.append(var_q)
            sigma_q: float | None = round(var_q ** 0.5, 5)
        else:
            sigma_q = None
        por_modelo[m] = {
            "sementes": [0, 1, 2],
            "ppl": [round(x, 4) for x in ppls],
            "media_ppl": round(media_ppl, 4),
            "dp_ppl": round(dp_ppl, 4),
            "cv_ppl_pct": round(100.0 * dp_ppl / media_ppl, 4) if media_ppl > 0 else None,
            "Q": [round(x, 4) for x in qs],
            "sigma_cal_Q": sigma_q,
            "Q_acc": [round(x, 4) for x in qas],
            "sigma_cal_Q_acc": round(float(np.std(qas, ddof=1)), 5) if len(qas) >= 2 else None,
        }
    sigma_comb = float(np.mean(vars_q) ** 0.5) if vars_q else 0.0
    return {
        "sigma_cal_combinado": round(sigma_comb, 5),
        "por_modelo": por_modelo,
    }


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
    saida["concordancia_q_acuracia"] = concordancia_acuracia(L)
    saida["replicas_awq"] = variancia_replicas_awq(L)
    saida["pareto"] = pareto(L)
    SAIDA.write_text(json.dumps(saida, ensure_ascii=False, indent=1, default=float))
    print(f"-> {SAIDA.relative_to(RAIZ.parent)}")


if __name__ == "__main__":
    main()
