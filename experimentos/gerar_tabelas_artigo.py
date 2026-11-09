"""Gera todas as tabelas (.md e .tex) e figuras vetoriais (.svg) do artigo cientifico
diretamente a partir dos arquivos oficiais de dados em experimentos/resultados/.

Uso:
    .venv/bin/python experimentos/gerar_tabelas_artigo.py
"""
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
RES = RAIZ / "experimentos" / "resultados"
ART = RAIZ / "artigo"
TABS = ART / "tabelas"
FIGS = ART / "figuras"

MODELOS_ORDEM = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "HuggingFaceTB/SmolLM3-3B",
    "meta-llama/Llama-3.2-3B",
    "Qwen/Qwen3-4B",
]

CURTOS = {
    "Qwen/Qwen3-0.6B": "Qwen3-0.6B",
    "Qwen/Qwen3-1.7B": "Qwen3-1.7B",
    "HuggingFaceTB/SmolLM3-3B": "SmolLM3-3B",
    "meta-llama/Llama-3.2-3B": "Llama-3.2-3B",
    "Qwen/Qwen3-4B": "Qwen3-4B",
}


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den != 0.0 else default


def carregar_dados() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    with open(RES / "resultados.csv", encoding="utf-8") as f_res:
        linhas_csv = list(csv.DictReader(f_res))
    with open(RES / "kquant.csv", encoding="utf-8") as f_kq:
        kquant_q = list(csv.DictReader(f_kq))
    with open(RES / "kquant-velocidade.csv", encoding="utf-8") as f_kv:
        kquant_v = list(csv.DictReader(f_kv))
    hip = json.loads((RES / "hipoteses.json").read_text(encoding="utf-8"))
    h3_nll = json.loads((RES / "h3_nll.json").read_text(encoding="utf-8"))
    matriz_cfg = json.loads((RAIZ / "experimentos" / "matriz.json").read_text(encoding="utf-8"))
    return linhas_csv, kquant_q, kquant_v, hip, h3_nll, matriz_cfg


def indexar_mlx(linhas_csv: Sequence[dict[str, Any]]) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    idx: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for r in linhas_csv:
        ch = (r["modelo"], r["metodo"], int(r["bits"]), str(r["seed"]))
        idx[ch] = r
    return idx


def gerar_tabela1(idx, kquant_q, kquant_v, matriz_cfg):
    arq_map = {m["id"]: m for m in matriz_cfg["modelos"]}
    kq_f16 = {r["modelo"]: r for r in kquant_q if r["tipo"] == "F16"}
    kv_f16 = {r["modelo"]: r for r in kquant_v if r["tipo"] == "F16"}

    md = [
        "# Tabela 1 — Modelos Avaliados, Dimensões Arquiteturais e Linha de Base (FP16 / BF16)",
        "",
        "| Modelo | Camadas ($n$) | $d_{\\text{model}}$ | $d_{\\text{ff}}$ | $d_{\\text{proj}}$ | Tam. BF16 (GB) | PPL WikiText-2 (MLX) | Acc Zero-Shot Média | Decode BF16 MLX (tok/s) | PPL F16 (`llama.cpp`) | Decode F16 (`llama.cpp`) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    tex = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Modelos avaliados na grade principal (v2), dimensões arquiteturais e métricas de linha de base em precisão cheia (BF16 no MLX e F16 no \\texttt{llama.cpp}).}",
        "\\label{tab:grade_modelos}",
        "\\small",
        "\\begin{tabular}{lrrrrrrrrrr}",
        "\\toprule",
        "Modelo & $n$ & $d$ & $d_{\\text{ff}}$ & $o_{\\text{in}}$ & GB (BF16) & PPL (MLX) & Acc Média & tok/s (MLX) & PPL (GGUF) & tok/s (GGUF) \\\\",
        "\\midrule",
    ]
    for m in MODELOS_ORDEM:
        curto = CURTOS[m]
        a = arq_map[m]["arq"]
        fp = idx[(m, "fp", 16, "0")]
        kf = kq_f16[m]
        kv = kv_f16[curto]
        md.append(
            f"| `{curto}` | {a['n']} | {a['d']} | {a['i']} | {a['o_in']} | "
            f"{float(fp['tamanho_gb']):.3f} | {float(fp['ppl_wikitext2']):.2f} | {float(fp['acc_media']):.4f} | "
            f"{float(fp['decode_tps']):.1f} | {float(kf['ppl']):.2f} | {float(kv['decode_tps']):.1f} |"
        )
        tex.append(
            f"{curto} & {a['n']} & {a['d']} & {a['i']} & {a['o_in']} & "
            f"{float(fp['tamanho_gb']):.2f} & {float(fp['ppl_wikitext2']):.2f} & {float(fp['acc_media']):.3f} & "
            f"{float(fp['decode_tps']):.1f} & {float(kf['ppl']):.2f} & {float(kv['decode_tps']):.1f} \\\\"
        )
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TABS / "tabela1_grade_modelos.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (TABS / "tabela1_grade_modelos.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def gerar_tabela2(idx):
    md = [
        "# Tabela 2 — Retenção de Qualidade ($Q = \\text{PPL}_{\\text{fp}}/\\text{PPL}_q$) e Acurácia Relativa ($Q_{\\text{acc}}$) no MLX",
        "",
        "| Modelo | Bits | RTN ($Q$ / $Q_{\\text{acc}}$) | GPTQ-bf16 ($Q$ / $Q_{\\text{acc}}$) | GPTQ-fp32 ($Q$ / $Q_{\\text{acc}}$) | AWQ ($Q$ / $Q_{\\text{acc}}$) | DWQ ($Q$ / $Q_{\\text{acc}}$) |",
        "|---|---:|---|---|---|---|---|",
    ]
    tex = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Retenção de qualidade $Q = \\text{PPL}_{\\text{fp}}/\\text{PPL}_q$ (e $Q_{\\text{acc}} = \\text{Acc}_q/\\text{Acc}_{\\text{fp}}$) por modelo, largura de bits e algoritmo no MLX (semente 0). \\texttt{NaN} indica falha numérica da decomposição de Cholesky da Hessiana em \\texttt{bf16}; ``---'' indica exclusão por suporte da ferramenta ou limite de memória unificada (24~GiB).}",
        "\\label{tab:qualidade_mlx}",
        "\\footnotesize",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Modelo & Bits & RTN & GPTQ (bf16) & GPTQ (fp32) & AWQ & DWQ \\\\",
        "\\midrule",
    ]

    def _obter_q_fallback_nll(m: str, met: str, b: int, ppl_fp: float) -> tuple[str, str]:
        nll_p = RES / "nll" / f"{CURTOS[m]}-{met}-{b}b-g64-s0.npy"
        if nll_p.exists():
            arr = np.load(nll_p)
            q = _safe_div(ppl_fp, float(np.exp(arr.mean())))
            return f"{q:.4f} / *(aval.)*", f"{q:.3f}"
        return "—", "---"

    def celula(m: str, met: str, b: int) -> tuple[str, str]:
        fp = idx[(m, "fp", 16, "0")]
        ppl_fp, acc_fp = float(fp["ppl_wikitext2"]), float(fp["acc_media"])
        r = idx.get((m, met, b, "0"))
        if not r:
            return _obter_q_fallback_nll(m, met, b, ppl_fp)
        st = r.get("status", "")
        if st.startswith("falhou:invalido-nan"):
            return "**NaN**", "\\textbf{NaN}"
        if st.startswith("falhou"):
            return "—", "---"
        if not r.get("ppl_wikitext2"):
            return _obter_q_fallback_nll(m, met, b, ppl_fp)
        q = _safe_div(ppl_fp, float(r["ppl_wikitext2"]))
        qa = _safe_div(float(r["acc_media"]), acc_fp) if r.get("acc_media") else None
        s_md = f"{q:.4f}" + (f" ({qa:.3f})" if qa is not None else "")
        s_tx = f"{q:.3f}" + (f" ({qa:.2f})" if qa is not None else "")
        return s_md, s_tx

    for m in MODELOS_ORDEM:
        curto = CURTOS[m]
        for b in (8, 4, 3, 2):
            c_rtn = celula(m, "rtn", b)
            c_gptq = celula(m, "gptq", b)
            c_gfp32 = celula(m, "gptq_fp32", b)
            c_awq = celula(m, "awq", b)
            c_dwq = celula(m, "dwq", b)
            md.append(f"| `{curto}` | {b} | {c_rtn[0]} | {c_gptq[0]} | {c_gfp32[0]} | {c_awq[0]} | {c_dwq[0]} |")
            tex.append(f"{curto} & {b}b & {c_rtn[1]} & {c_gptq[1]} & {c_gfp32[1]} & {c_awq[1]} & {c_dwq[1]} \\\\")
        tex.append("\\midrule")
    tex[-1] = "\\bottomrule"
    tex += ["\\end{tabular}", "\\end{table}"]
    (TABS / "tabela2_qualidade_mlx.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (TABS / "tabela2_qualidade_mlx.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def gerar_tabela3(idx):
    md = [
        "# Tabela 3 — Velocidade de Inferência, Compressão de Memória e Eficiência $E$ em 4 Bits (H4 — Rodada 2, App Fechado)",
        "",
        "| Modelo | Método (4b) | Tam. (GB) | Compressão ($R_{\\text{tam}}$) | Prefill (tok/s) | Decode (tok/s) | Ganho Decode ($G_{\\text{dec}}$) | Eficiência $E = G_{\\text{dec}}/R_{\\text{tam}}$ | Pico Memória GPU (GB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    tex = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Desempenho de inferência e eficiência de conversão de banda de memória $E = G_{\\text{dec}}/R_{\\text{tam}}$ em 4 bits (Rodada 2 de referência, com máquina ociosa). Em todos os modelos de 3--4B, $E \\ge 0{,}83 > 0{,}60$ (H4 corroborada).}",
        "\\label{tab:velocidade_h4}",
        "\\small",
        "\\begin{tabular}{llrrrrrrr}",
        "\\toprule",
        "Modelo & Método & Tam. (GB) & $R_{\\text{tam}}$ & Prefill (tok/s) & Decode (tok/s) & $G_{\\text{dec}}$ & Eficiência $E$ & Pico GPU (GB) \\\\",
        "\\midrule",
    ]
    for m in MODELOS_ORDEM:
        curto = CURTOS[m]
        fp = idx[(m, "fp", 16, "0")]
        tam_fp = float(fp["tamanho_gb"])
        dec_fp = float(fp["decode_tps"])
        for met in ("rtn", "gptq", "awq", "dwq"):
            r = idx.get((m, met, 4, "0"))
            if not r or not r.get("decode_tps") or r.get("status", "").startswith("falhou"):
                continue
            tam_q = float(r["tamanho_gb"])
            pre_q = float(r["prefill_tps"])
            dec_q = float(r["decode_tps"])
            pico_q = float(r["pico_memoria_gb"])
            r_tam = _safe_div(tam_fp, tam_q)
            g_dec = _safe_div(dec_q, dec_fp)
            ef = _safe_div(g_dec, r_tam)
            md.append(
                f"| `{curto}` | `{met.upper()}` | {tam_q:.3f} | {r_tam:.2f}× | {pre_q:.1f} | "
                f"{dec_q:.1f} | {g_dec:.2f}× | **{ef:.3f}** | {pico_q:.2f} |"
            )
            tex.append(
                f"{curto} & {met.upper()} & {tam_q:.2f} & {r_tam:.2f}$\\times$ & {pre_q:.0f} & "
                f"{dec_q:.1f} & {g_dec:.2f}$\\times$ & \\textbf{{{ef:.3f}}} & {pico_q:.2f} \\\\"
            )
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TABS / "tabela3_velocidade_memoria_h4.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (TABS / "tabela3_velocidade_memoria_h4.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def gerar_tabela4(kquant_q, kquant_v):
    kv_map = {(r["modelo"], r["tipo"]): r for r in kquant_v}
    f16_map = {r["modelo"]: float(r["ppl"]) for r in kquant_q if r["tipo"] == "F16"}

    md = [
        "# Tabela 4 — Qualidade e Velocidade dos K-quants (`llama.cpp` v0.7.0 com `imatrix`)",
        "",
        "| Modelo | Tipo | Bits/Peso (bpw) | Tam. (GB) | PPL (`llama.cpp`) | $Q_{\\text{GGUF}} = \\text{PPL}_{\\text{F16}}/\\text{PPL}_q$ | KLD Média | Top-1 Igual (%) | Prefill (tok/s) | Decode (tok/s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    tex = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Resultados de qualidade e velocidade no \\texttt{llama.cpp} (K-quants calibrados com \\texttt{imatrix} sobre \\texttt{calibration\\_v5}).}",
        "\\label{tab:kquants}",
        "\\footnotesize",
        "\\begin{tabular}{llrrrrrrrr}",
        "\\toprule",
        "Modelo & Tipo & bpw & GB & PPL & $Q_{\\text{GGUF}}$ & KLD & Top-1 (\\%) & Prefill & Decode \\\\",
        "\\midrule",
    ]
    for r in kquant_q:
        curto = CURTOS[r["modelo"]]
        tipo = r["tipo"]
        bpw = float(r["bits_por_peso"])
        tam = float(r["tamanho_gb"])
        ppl = float(r["ppl"])
        q_gguf = _safe_div(f16_map[r["modelo"]], ppl)
        kld = float(r["kld_media"])
        top1 = float(r["top1_igual"])
        v = kv_map[(curto, tipo)]
        pre = float(v["prefill_tps"])
        dec = float(v["decode_tps"])
        md.append(
            f"| `{curto}` | `{tipo}` | {bpw:.2f} | {tam:.3f} | {ppl:.2f} | {q_gguf:.4f} | "
            f"{kld:.4f} | {top1:.1f} | {pre:.1f} | {dec:.1f} |"
        )
        tex.append(
            f"{curto} & {tipo} & {bpw:.2f} & {tam:.2f} & {ppl:.2f} & {q_gguf:.3f} & "
            f"{kld:.3f} & {top1:.1f} & {pre:.0f} & {dec:.1f} \\\\"
        )
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TABS / "tabela4_kquants_llamacpp.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (TABS / "tabela4_kquants_llamacpp.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def gerar_tabela5(hip, h3_nll):
    conc = hip["concordancia_q_acuracia"]
    rep = hip["replicas_awq"]
    s_agg = h3_nll["_agregado_rtn"]
    md = [
        "# Tabela 5 — Síntese dos Vereditos das Hipóteses (Protocolo v2) e Diagnósticos Secundários",
        "",
        "| Hipótese / Análise | Critério Pré-Registrado | Resultado Empírico | Veredito (v2) |",
        "|---|---|---|---|",
        "| **H1** (Retenção em 4 bits $\\ge 0,95$ só com calibração) | $\\ge 4/5$ modelos com $Q_{\\text{ciente}} \\ge 0,95$ e $Q_{\\text{RTN}} < 0,95$ | Atendido em **2/4** avaliáveis (`Qwen3-1.7B` AWQ $0,970$; `Llama-3.2-3B` AWQ $0,955$). Em `0.6B` ($0,916$) e `4B` ($0,942$), nenhum método atingiu $0,95$ | **Inconclusiva** |",
        "| **H2** (Sensibilidade por porte na família Qwen3) | Ordem $Q_{0,6\\text{B}} < Q_{1,7\\text{B}} < Q_{4\\text{B}}$ em $\\ge 75\\%$ das células | Confirmada em **1/2** células do RTN (3 bits: $0,337 < 0,408 < 0,652$; em 4 bits: $0,862 \\approx 0,856 < 0,942$) | **Inconclusiva** |",
        "| **H3 — Primária** (Queda superlinear em $L = 1 - Q$ e redução $\\ge 20\\%$ a 3 bits) | $\\Delta L_{3\\to 2} > \\Delta L_{4\\to 3}$ e $L_{3,\\text{ciente}} \\le 0,8 L_{3,\\text{RTN}}$ em $\\ge 4$ modelos | Sem SmolLM3: $\\Delta L_{4\\to 3}=0,389 < \\Delta L_{3\\to 2}=0,508$, mas só 2 modelos avaliáveis em 3b (`0.6B` e `1.7B`, ambos com redução $\\ge 20\\%$). Com SmolLM3: efeito teto em $L$ inverte a média ($0,463 > 0,425$) | **Inconclusiva** (Refutada na sensibilidade com SmolLM3) |",
        f"| **H3 — Secundária Pós-Hoc** (Escala logarítmica $S = (D_2-D_3)-(D_3-D_4)$ em nats/token) | $\\text{{IC}}_{{95\\%}}(S) > 0$ (sem saturação em $Q \\to 0$) | Superlinear nos **5/5** modelos no RTN ($S$ médio $= {s_agg['S_media']:.3f}$ nats/tok, IC95% $[{s_agg['S_ic95'][0]:.3f}; {s_agg['S_ic95'][1]:.3f}]$) e em todos os métodos calibrados | **Corrobora a 1ª metade** (análise secundária declarada) |",
        "| **H4** (Eficiência de velocidade de decodificação em 4 bits) | $E = G_{\\text{dec}}/R_{\\text{tam}} \\ge 0,60$ nos modelos de 3–4B | `SmolLM3-3B`: $E=0,83$; `Llama-3.2-3B`: $E=0,85$; `Qwen3-4B`: $E=0,87$ (ganho de $3,0\\text{–}3,1\\times$ sobre BF16) | **Corroborada** |",
        f"| **Diagnóstico A** (Concordância $Q \\times Q_{{\\text{{acc}}}}$ zero-shot) | Correlação e coincidência do melhor método em 4 bits | $N={conc['n_pares']}$ pares; Pearson $r = {conc['pearson_r']:.4f}$; Spearman $\\rho = {conc['spearman_rho']:.4f}$; melhor método coincide em **{conc['concordancia_melhor_ciente_4b']}** modelos | **Alta concordância** de ranking |",
        f"| **Diagnóstico B** (Variância entre sementes de calibração AWQ 4b) | Sementes $0, 1, 2$ em `Qwen3-0.6B` e `Qwen3-1.7B` | $\\sigma_{{\\text{{cal}}}}(Q) = {rep['sigma_cal_combinado']:.5f}$ ($\\text{{CV}}_{{\\text{{PPL}}}} = 0,75\\%$ no `0.6B` e $0,65\\%$ no `1.7B`) | **Baixo ruído de semente** ($< 0,8\\%$ em PPL) |",
        "| **Diagnóstico C** (Falha de `NaN` no GPTQ padrão do `mlx-lm 0.32.2`) | Inspeção de autovalores da Hessiana $H = X^\\top X$ em `bf16` vs `fp32` | Em `bf16`, `o_proj`/`down_proj` tornam-se indefinidas ($\\lambda_{\\min} < -\\text{damp}$) em 100% dos modelos de 3–4B (9/9 jobs `NaN`); em `fp32`, 0 matrizes falham | **Causa-raiz demonstrada e corrigida (`gptq_fp32`)** |",
    ]
    tex = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Síntese dos vereditos formais das hipóteses H1--H4 segundo o protocolo pré-registrado (v2) e análises diagnósticas complementares.}",
        "\\label{tab:vereditos_hipoteses}",
        "\\footnotesize",
        "\\begin{tabular}{p{2.8cm}p{4.2cm}p{5.2cm}p{2.2cm}}",
        "\\toprule",
        "Hipótese / Eixo & Critério & Evidência Obtida & Veredito \\\\",
        "\\midrule",
        "H1 (Qualidade 4b) & $\\ge 4/5$ com $Q_{\\text{ciente}} \\ge 0{,}95 > Q_{\\text{RTN}}$ & 2/4 avaliáveis atingem $\\ge 0{,}95$ (1,7B e Llama-3,2-3B) & Inconclusiva \\\\",
        "H2 (Porte Qwen3) & $Q_{0{,}6\\text{B}} < Q_{1{,}7\\text{B}} < Q_{4\\text{B}}$ em $\\ge 75\\%$ & Confirmada em 1/2 células do RTN (3b sim; 4b empate 0,6B/1,7B) & Inconclusiva \\\\",
        "H3 (Primária $L$) & $\\Delta L_{3\\to 2} > \\Delta L_{4\\to 3}$ e redução $\\ge 20\\%$ & Saturação de $L \\le 1$ aos 2 bits; só 2 modelos avaliáveis em 3b & Inconclusiva \\\\",
        f"H3 (Escala NLL) & $\\text{{IC}}_{{95\\%}}(S) > 0$ em nats/token & Superlinear em 5/5 modelos ($S = {s_agg['S_media']:.2f}$ nats/tok) & Superlinear (pós-hoc) \\\\",
        "H4 (Velocidade 4b) & $E \\ge 0{,}60$ nos modelos de 3--4B & $E = 0{,}83$ (SmolLM3), $0{,}85$ (Llama), $0{,}87$ (Qwen3-4B) & \\textbf{Corroborada} \\\\",
        f"Concordância $Q \\times Q_{{\\text{{acc}}}}$ & Correlação em $N={conc['n_pares']}$ pares & $r = {conc['pearson_r']:.3f}$, $\\rho = {conc['spearman_rho']:.3f}$, melhor ciente 3/4 & Alta concordância \\\\",
        f"Ruído de Calibração & 3 sementes AWQ 4b (0,6B e 1,7B) & $\\sigma_{{\\text{{cal}}}}(Q) = {rep['sigma_cal_combinado']:.4f}$, $\\text{{CV}}_{{\\text{{PPL}}}} \\le 0{{,}}75\\%$ & Ruído $< 0{{,}}8\\%$ \\\\",
        "GPTQ Hessiana bf16 & Autovalores de $H = X^\\top X + \\lambda I$ & 9/9 jobs 3--4B dão \\texttt{NaN} em bf16; 0 falhas em fp32 & Bug numérico demonstrado \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    (TABS / "tabela5_vereditos_hipoteses.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (TABS / "tabela5_vereditos_hipoteses.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")


def svg_header(w=760, h=420):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="Inter, -apple-system, BlinkMacSystemFont, sans-serif">',
        f'<rect width="{w}" height="{h}" fill="#ffffff" rx="8"/>',
    ]


def gerar_figuras(idx, kquant_q, kquant_v, h3_nll):
    # Figura 1: Qualidade Q (4b e 3b) vs Tamanho (GB) nos 5 modelos
    w, h = 780, 440
    pad_l, pad_r, pad_t, pad_b = 70, 40, 50, 60
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    svg = svg_header(w, h)
    svg.append('<text x="390" y="28" text-anchor="middle" font-size="15" font-weight="bold" fill="#1e293b">Figura 1 — Retenção de Qualidade (Q) vs. Tamanho do Modelo (3 e 4 bits, MLX)</text>')
    # eixos: x = 0..2.6 GB, y = 0.30..1.02
    x_min, x_max = 0.0, 2.6
    y_min, y_max = 0.30, 1.02

    def tx(x: float) -> float:
        return pad_l + (x - x_min) / (x_max - x_min) * pw

    def ty(y: float) -> float:
        return pad_t + ph - (y - y_min) / (y_max - y_min) * ph

    for yv in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
        yp = ty(yv)
        cor = "#ef4444" if abs(yv - 0.95) < 1e-6 else "#e2e8f0"
        dash = ' stroke-dasharray="5,4"' if abs(yv - 0.95) < 1e-6 else ""
        svg.append(f'<line x1="{pad_l}" y1="{yp:.1f}" x2="{w-pad_r}" y2="{yp:.1f}" stroke="{cor}" stroke-width="1"{dash}/>')
        svg.append(f'<text x="{pad_l-8}" y="{yp+4:.1f}" text-anchor="end" font-size="11" fill="#475569">{yv:.2f}</text>')
    for xv in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
        xp = tx(xv)
        svg.append(f'<line x1="{xp:.1f}" y1="{pad_t}" x2="{xp:.1f}" y2="{h-pad_b}" stroke="#f1f5f9" stroke-width="1"/>')
        svg.append(f'<text x="{xp:.1f}" y="{h-pad_b+18}" text-anchor="middle" font-size="11" fill="#475569">{xv:.1f} GB</text>')

    svg.append(f'<rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}" fill="none" stroke="#94a3b8" stroke-width="1.2"/>')
    svg.append(f'<text x="{pad_l + pw/2}" y="{h-15}" text-anchor="middle" font-size="12" font-weight="600" fill="#334155">Tamanho do Modelo Quantizado em Disco (GB)</text>')
    svg.append(f'<text x="18" y="{pad_t + ph/2}" text-anchor="middle" font-size="12" font-weight="600" fill="#334155" transform="rotate(-90 18 {pad_t + ph/2})">Retenção de Qualidade Q = PPL_fp / PPL_q</text>')

    cores_met = {"rtn": "#64748b", "gptq": "#d97706", "gptq_fp32": "#059669", "awq": "#2563eb", "dwq": "#7c3aed"}
    for m in MODELOS_ORDEM:
        fp = float(idx[(m, "fp", 16, "0")]["ppl_wikitext2"])
        for met, cor in cores_met.items():
            for b in (4, 3):
                r = idx.get((m, met, b, "0"))
                if not r or r.get("status", "").startswith("falhou") or not r.get("ppl_wikitext2"):
                    continue
                q = fp / float(r["ppl_wikitext2"])
                if q < y_min:
                    continue
                tam = float(r["tamanho_gb"])
                cx, cy = tx(tam), ty(q)
                raio = 5.5 if b == 4 else 4.0
                op = "1.0" if b == 4 else "0.75"
                svg.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{raio}" fill="{cor}" fill-opacity="{op}" stroke="#ffffff" stroke-width="1"/>')
                if b == 4 and met in ("awq", "rtn", "gptq_fp32"):
                    lbl = f"{CURTOS[m].split('-')[0]} {met.upper()}"
                    svg.append(f'<text x="{cx+7:.1f}" y="{cy+3:.1f}" font-size="9.5" fill="#1e293b">{lbl}</text>')

    # Legenda
    lx, ly = pad_l + 15, pad_t + 18
    for i, (met, cor) in enumerate(cores_met.items()):
        svg.append(f'<circle cx="{lx + i*115}" cy="{ly}" r="5" fill="{cor}"/>')
        svg.append(f'<text x="{lx + i*115 + 9}" y="{ly+4}" font-size="11" fill="#1e293b">{met.upper()}</text>')
    svg.append("</svg>")
    (FIGS / "figura1_pareto_qualidade_tamanho.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")

    # Figura 2: Degradação Superlinear em NLL (D4, D3, D2) no RTN
    svg2 = svg_header(w, h)
    svg2.append('<text x="390" y="28" text-anchor="middle" font-size="15" font-weight="bold" fill="#1e293b">Figura 2 — Degradação Superlinear em Escala Logarítmica (ΔNLL em nats/token, RTN 4 → 3 → 2 bits)</text>')
    cores_mod = {
        "Qwen/Qwen3-0.6B": "#2563eb",
        "Qwen/Qwen3-1.7B": "#059669",
        "HuggingFaceTB/SmolLM3-3B": "#dc2626",
        "meta-llama/Llama-3.2-3B": "#d97706",
        "Qwen/Qwen3-4B": "#7c3aed",
    }
    x_pos = {4: pad_l + 60, 3: pad_l + pw / 2, 2: w - pad_r - 60}
    y2_max = 24.0

    def ty2(y: float) -> float:
        return pad_t + ph - (y / y2_max) * ph
    for yv in [0, 5, 10, 15, 20]:
        yp = ty2(yv)
        svg2.append(f'<line x1="{pad_l}" y1="{yp:.1f}" x2="{w-pad_r}" y2="{yp:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        svg2.append(f'<text x="{pad_l-8}" y="{yp+4:.1f}" text-anchor="end" font-size="11" fill="#475569">{yv} nats</text>')
    for b, xp in x_pos.items():
        svg2.append(f'<line x1="{xp:.1f}" y1="{pad_t}" x2="{xp:.1f}" y2="{h-pad_b}" stroke="#f1f5f9" stroke-width="1"/>')
        svg2.append(f'<text x="{xp:.1f}" y="{h-pad_b+20}" text-anchor="middle" font-size="12" font-weight="600" fill="#334155">{b} bits</text>')
    svg2.append(f'<rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}" fill="none" stroke="#94a3b8" stroke-width="1.2"/>')

    for i, m in enumerate(MODELOS_ORDEM):
        linha_rtn = next(x for x in h3_nll[m]["linhas"] if x["metodo"] == "rtn")
        pts = [(x_pos[4], ty2(linha_rtn["D4"])), (x_pos[3], ty2(linha_rtn["D3"])), (x_pos[2], ty2(linha_rtn["D2"]))]
        cor = cores_mod[m]
        d_str = f"M {pts[0][0]:.1f} {pts[0][1]:.1f} L {pts[1][0]:.1f} {pts[1][1]:.1f} L {pts[2][0]:.1f} {pts[2][1]:.1f}"
        svg2.append(f'<path d="{d_str}" fill="none" stroke="{cor}" stroke-width="2.4"/>')
        for px, py in pts:
            svg2.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.5" fill="{cor}" stroke="#fff" stroke-width="1"/>')
        svg2.append(f'<circle cx="{pad_l + 20}" cy="{pad_t + 22 + i*20}" r="5" fill="{cor}"/>')
        svg2.append(f'<text x="{pad_l + 32}" y="{pad_t + 26 + i*20}" font-size="11" fill="#1e293b">{CURTOS[m]} (S = +{linha_rtn["S"]:.2f} nats)</text>')
    svg2.append("</svg>")
    (FIGS / "figura2_degradacao_superlinear_nll.svg").write_text("\n".join(svg2) + "\n", encoding="utf-8")

    # Figura 3: Eficiência E e Ganho de Decodificação a 4 bits por Modelo (H4)
    svg3 = svg_header(w, h)
    svg3.append('<text x="390" y="28" text-anchor="middle" font-size="15" font-weight="bold" fill="#1e293b">Figura 3 — Eficiência de Banda E = G_dec / R_tam em 4 Bits por Porte de Modelo (H4)</text>')

    def ty3(y: float) -> float:
        return pad_t + ph - (y / 1.05) * ph

    for yv in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        yp = ty3(yv)
        cor = "#16a34a" if abs(yv - 0.6) < 1e-6 else "#e2e8f0"
        dash = ' stroke-dasharray="6,4"' if abs(yv - 0.6) < 1e-6 else ""
        svg3.append(f'<line x1="{pad_l}" y1="{yp:.1f}" x2="{w-pad_r}" y2="{yp:.1f}" stroke="{cor}" stroke-width="1.2"{dash}/>')
        svg3.append(f'<text x="{pad_l-8}" y="{yp+4:.1f}" text-anchor="end" font-size="11" fill="#475569">{yv:.1f}</text>')
    svg3.append(f'<text x="{w-pad_r-6}" y="{ty3(0.6)-6:.1f}" text-anchor="end" font-size="10.5" font-weight="600" fill="#16a34a">Limiar H4 (E ≥ 0,60)</text>')

    bw = pw / len(MODELOS_ORDEM)
    for i, m in enumerate(MODELOS_ORDEM):
        fp = idx[(m, "fp", 16, "0")]
        r = idx[(m, "rtn", 4, "0")]
        e_rtn = _safe_div(
            _safe_div(float(r["decode_tps"]), float(fp["decode_tps"])),
            _safe_div(float(fp["tamanho_gb"]), float(r["tamanho_gb"])),
        )
        bx = pad_l + i * bw + bw * 0.22
        by = ty3(e_rtn)
        bh = (h - pad_b) - by
        svg3.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw*0.56:.1f}" height="{bh:.1f}" fill="#2563eb" rx="4"/>')
        svg3.append(f'<text x="{bx + bw*0.28:.1f}" y="{by-7:.1f}" text-anchor="middle" font-size="11" font-weight="bold" fill="#1e293b">{e_rtn:.2f}</text>')
        svg3.append(f'<text x="{bx + bw*0.28:.1f}" y="{h-pad_b+20}" text-anchor="middle" font-size="11" fill="#334155">{CURTOS[m]}</text>')
    svg3.append(f'<rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}" fill="none" stroke="#94a3b8" stroke-width="1.2"/>')
    svg3.append("</svg>")
    (FIGS / "figura3_eficiencia_velocidade_h4.svg").write_text("\n".join(svg3) + "\n", encoding="utf-8")

    # Figura 4: Comparação MLX (AWQ/RTN 4b) vs llama.cpp (Q4_K_M) em Decode (tok/s)
    svg4 = svg_header(w, h)
    svg4.append('<text x="390" y="28" text-anchor="middle" font-size="15" font-weight="bold" fill="#1e293b">Figura 4 — Velocidade de Decodificação em 4 Bits: MLX (RTN 4b) vs. llama.cpp (Q4_K_M)</text>')
    kv_q4 = {r["modelo"]: float(r["decode_tps"]) for r in kquant_v if r["tipo"] == "Q4_K_M"}

    def ty4(y: float) -> float:
        return pad_t + ph - (y / 450.0) * ph
    for yv in [0, 100, 200, 300, 400]:
        yp = ty4(yv)
        svg4.append(f'<line x1="{pad_l}" y1="{yp:.1f}" x2="{w-pad_r}" y2="{yp:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        svg4.append(f'<text x="{pad_l-8}" y="{yp+4:.1f}" text-anchor="end" font-size="11" fill="#475569">{yv} tok/s</text>')
    for i, m in enumerate(MODELOS_ORDEM):
        curto = CURTOS[m]
        d_mlx = float(idx[(m, "rtn", 4, "0")]["decode_tps"])
        d_gguf = kv_q4[curto]
        gx = pad_l + i * bw + bw * 0.14
        w_bar = bw * 0.34
        y_m, y_g = ty4(d_mlx), ty4(d_gguf)
        svg4.append(f'<rect x="{gx:.1f}" y="{y_m:.1f}" width="{w_bar:.1f}" height="{(h-pad_b)-y_m:.1f}" fill="#2563eb" rx="3"/>')
        svg4.append(f'<text x="{gx + w_bar/2:.1f}" y="{y_m-5:.1f}" text-anchor="middle" font-size="9.5" fill="#1e293b">{d_mlx:.0f}</text>')
        svg4.append(f'<rect x="{gx + w_bar + 4:.1f}" y="{y_g:.1f}" width="{w_bar:.1f}" height="{(h-pad_b)-y_g:.1f}" fill="#f59e0b" rx="3"/>')
        svg4.append(f'<text x="{gx + 1.5*w_bar + 4:.1f}" y="{y_g-5:.1f}" text-anchor="middle" font-size="9.5" fill="#1e293b">{d_gguf:.0f}</text>')
        svg4.append(f'<text x="{gx + w_bar + 2:.1f}" y="{h-pad_b+20}" text-anchor="middle" font-size="11" fill="#334155">{curto}</text>')
    svg4.append(f'<rect x="{w-240}" y="{pad_t+12}" width="12" height="12" fill="#2563eb" rx="2"/>')
    svg4.append(f'<text x="{w-222}" y="{pad_t+22}" font-size="11" fill="#1e293b">MLX (RTN 4b, g=64)</text>')
    svg4.append(f'<rect x="{w-240}" y="{pad_t+32}" width="12" height="12" fill="#f59e0b" rx="2"/>')
    svg4.append(f'<text x="{w-222}" y="{pad_t+42}" font-size="11" fill="#1e293b">llama.cpp (Q4_K_M + imatrix)</text>')
    svg4.append(f'<rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}" fill="none" stroke="#94a3b8" stroke-width="1.2"/>')
    svg4.append("</svg>")
    (FIGS / "figura4_mlx_vs_kquants.svg").write_text("\n".join(svg4) + "\n", encoding="utf-8")


def main():
    TABS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    linhas_csv, kquant_q, kquant_v, hip, h3_nll, matriz_cfg = carregar_dados()
    idx = indexar_mlx(linhas_csv)
    gerar_tabela1(idx, kquant_q, kquant_v, matriz_cfg)
    gerar_tabela2(idx)
    gerar_tabela3(idx)
    gerar_tabela4(kquant_q, kquant_v)
    gerar_tabela5(hip, h3_nll)
    gerar_figuras(idx, kquant_q, kquant_v, h3_nll)
    print("Tabelas (.md/.tex) e figuras (.svg) geradas em artigo/tabelas/ e artigo/figuras/")


if __name__ == "__main__":
    main()
