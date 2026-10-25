"""Analise SECUNDARIA (pos-hoc, protocolo-v2) da H3 em escala logaritmica.

  python experimentos/analise_h3_nll.py

Motivo: Q = PPL_fp/PPL_q satura em 0 aos 2 bits, e o incremento de L = 1 - Q de 3 para 2 bits tem teto igual ao
Q de 3 bits; o criterio pre-registrado da H3 ("incrementos de L crescentes") falha por construcao sempre que a
perda de 4 para 3 bits passa de ~0,5. Aqui a perda e medida em nats por token, sem teto:

    D_b = media_janelas(NLL_q,b - NLL_fp) = ln(PPL_q / PPL_fp)
    S   = (D_2 - D_3) - (D_3 - D_4)          superlinear se IC95% de S > 0

IC por bootstrap PAREADO sobre as janelas de 2048 tokens (as mesmas janelas em fp e nas tres larguras de bit),
10 000 reamostragens, mesma semente da analise.py. Tambem reproduz, sem alteracao, o criterio original (L) e a
segunda metade da H3 (reducao de L >= 20% em 3 bits do melhor entre GPTQ e DWQ disponiveis contra o RTN).
NAO substitui o veredito de analise.py: e uma analise complementar, declarada como pos-hoc.
Linhas com status "falhou*" (inclusive o GPTQ invalido do SmolLM3) ficam de fora, como em analise.py.
"""
import csv
import json
from pathlib import Path

import numpy as np

from analise import CIENTES, MODELOS, ident

RAIZ = Path(__file__).resolve().parent
RES = RAIZ / "resultados"
CSV, NLL = RES / "resultados.csv", RES / "nll"
B = 10_000
rng = np.random.default_rng(0)
METODOS = ["rtn", *CIENTES]


def carregar():
    L = [r for r in csv.DictReader(open(CSV)) if r.get("ppl_wikitext2") and not r.get("status", "").startswith("falhou")]
    fp = {r["modelo"]: r for r in L if r["metodo"] == "fp"}
    q = {(r["modelo"], r["metodo"], int(r["bits"])): r for r in L if r["metodo"] != "fp" and r["seed"] == "0"}
    return fp, q


def janelas(r):
    p = NLL / f"{ident(r)}.npy"
    return np.load(p).astype(np.float64) if p.exists() else None


def ic(x):
    return [round(float(v), 3) for v in np.percentile(x, [2.5, 97.5])]


def por_modelo(m, fp, q):
    """Estatisticas por metodo para um modelo; devolve (linhas, S_boot_rtn, info_criterio_original)."""
    nf = janelas(fp[m])
    if nf is None:
        return [], None, None
    idx = rng.integers(0, len(nf), (B, len(nf)))          # mesmas janelas para todos os metodos e larguras
    delta = {}
    for met in METODOS:
        for b in (4, 3, 2):
            r = q.get((m, met, b))
            nq = janelas(r) if r else None
            if nq is not None and len(nq) == len(nf):
                d = nq - nf
                delta[(met, b)] = (d.mean(), d[idx].mean(1))
    linhas, s_rtn = [], None
    for met in METODOS:
        if not all((met, b) in delta for b in (4, 3, 2)):
            faltam = [b for b in (4, 3, 2) if (met, b) not in delta]
            if any((met, b) in delta for b in (4, 3, 2)):
                linhas.append(dict(modelo=m, metodo=met, nota=f"sem {faltam} bits: nao se calcula o incremento"))
            continue
        (d4, b4), (d3, b3), (d2, b2) = (delta[(met, b)] for b in (4, 3, 2))
        s, sb = (d2 - d3) - (d3 - d4), (b2 - b3) - (b3 - b4)
        lo, hi = ic(sb)
        linhas.append(dict(modelo=m, metodo=met, D4=round(d4, 3), D3=round(d3, 3), D2=round(d2, 3),
                           inc_4_3=round(d3 - d4, 3), inc_3_2=round(d2 - d3, 3), S=round(s, 3), S_ic95=[lo, hi],
                           superlinear=("sim" if lo > 0 else "nao" if hi < 0 else "inconclusivo")))
        if met == "rtn":
            s_rtn = sb
    # criterio original (L = 1 - Q) para o RTN, sem alteracao
    orig = None
    if all(("rtn", b) in delta for b in (4, 3, 2)):
        Lb = {b: 1 - np.exp(-delta[("rtn", b)][0]) for b in (4, 3, 2)}
        orig = dict(L4=round(Lb[4], 3), L3=round(Lb[3], 3), L2=round(Lb[2], 3),
                    inc_4_3=round(Lb[3] - Lb[4], 3), inc_3_2=round(Lb[2] - Lb[3], 3), teto_inc_3_2=round(1 - Lb[3], 3))
    # segunda metade: reducao de L em 3 bits (melhor entre GPTQ e DWQ disponiveis) contra o RTN
    red = None
    cand = [met for met in ("gptq", "dwq") if (met, 3) in delta]
    if ("rtn", 3) in delta and cand:
        Lq = lambda met: 1 - np.exp(-delta[(met, 3)][0])
        melhor = min(cand, key=Lq)
        razao = lambda met: (1 - np.exp(-delta[(met, 3)][1])) / (1 - np.exp(-delta[("rtn", 3)][1]))
        rb = razao(melhor)
        red = dict(melhor=melhor, candidatos=cand, razao_L=round(float(Lq(melhor) / (1 - np.exp(-delta[("rtn", 3)][0]))), 3),
                   razao_ic95=ic(rb), reduz_20pct=bool(Lq(melhor) <= 0.8 * (1 - np.exp(-delta[("rtn", 3)][0]))))
    return linhas, s_rtn, dict(original=orig, reducao_3b=red)


def main():
    fp, q = carregar()
    saida, s_modelos, tabela = {}, {}, []
    for m in MODELOS:
        if m not in fp:
            continue
        linhas, s_rtn, crit = por_modelo(m, fp, q)
        if not linhas and crit is None:
            continue
        saida[m] = dict(linhas=linhas, criterio=crit)
        tabela += linhas
        if s_rtn is not None:
            s_modelos[m] = s_rtn
    agg = None
    if s_modelos:
        media = np.mean(np.stack(list(s_modelos.values())), axis=0)   # modelos independentes: reamostragem independente
        agg = dict(n_modelos=len(s_modelos), modelos=list(s_modelos), S_media=round(float(media.mean()), 3), S_ic95=ic(media))
    saida["_agregado_rtn"] = agg

    md = ["# H3 em escala log (analise secundaria, pos-hoc)", "",
          "D = ln(PPL_q/PPL_fp) em nats/token; S = (D2-D3)-(D3-D4); superlinear se IC95% de S > 0.", "",
          "| Modelo | Metodo | D4 | D3 | D2 | 4->3 | 3->2 | S | IC95% de S | Superlinear |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in tabela:
        if "nota" in r:
            md.append(f"| {r['modelo'].split('/')[-1]} | {r['metodo']} | | | | | | | {r['nota']} | |")
        else:
            md.append(f"| {r['modelo'].split('/')[-1]} | {r['metodo']} | {r['D4']} | {r['D3']} | {r['D2']} | {r['inc_4_3']} | "
                      f"{r['inc_3_2']} | {r['S']} | {r['S_ic95']} | {r['superlinear']} |")
    if agg:
        md += ["", f"**RTN, media de S sobre {agg['n_modelos']} modelo(s):** {agg['S_media']} (IC95% {agg['S_ic95']}).", ""]
    md += ["## Criterio original (L = 1 - Q), RTN, sem alteracao", "",
           "| Modelo | L4 | L3 | L2 | 4->3 | 3->2 | Teto do 3->2 |", "|---|---|---|---|---|---|---|"]
    for m, v in saida.items():
        o = (v.get("criterio") or {}).get("original") if m != "_agregado_rtn" else None
        if o:
            md.append(f"| {m.split('/')[-1]} | {o['L4']} | {o['L3']} | {o['L2']} | {o['inc_4_3']} | {o['inc_3_2']} | {o['teto_inc_3_2']} |")
    md += ["", "## Segunda metade: reducao de L em 3 bits (melhor entre GPTQ e DWQ disponiveis) contra o RTN", "",
           "| Modelo | Melhor | L_melhor/L_RTN | IC95% | Reduz >= 20% |", "|---|---|---|---|---|"]
    for m, v in saida.items():
        r = (v.get("criterio") or {}).get("reducao_3b") if m != "_agregado_rtn" else None
        if r:
            md.append(f"| {m.split('/')[-1]} | {r['melhor']} (de {r['candidatos']}) | {r['razao_L']} | {r['razao_ic95']} | {'sim' if r['reduz_20pct'] else 'nao'} |")
    (RES / "h3_nll.md").write_text("\n".join(md) + "\n")
    (RES / "h3_nll.json").write_text(json.dumps(saida, ensure_ascii=False, indent=1, default=float))
    print("\n".join(md))
    print(f"\n-> {(RES / 'h3_nll.md').relative_to(RAIZ.parent)}, {(RES / 'h3_nll.json').relative_to(RAIZ.parent)}")


if __name__ == "__main__":
    main()
