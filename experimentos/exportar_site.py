"""Gera site/data/resultados.json a partir de resultados.csv (e hipoteses.json, se existir).

Tolerante a colunas novas ou ausentes: colunas desconhecidas são preservadas
(números viram float, vazio vira null). Uso:
    .venv/bin/python experimentos/exportar_site.py
"""
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
CSV = RAIZ / "experimentos" / "resultados" / "resultados.csv"
PILOTO = RAIZ / "experimentos" / "resultados" / "piloto-v0.csv"  # usado enquanto a matriz nao gerou resultados.csv
KQUANT = RAIZ / "experimentos" / "resultados" / "kquant.csv"
HIPOTESES = RAIZ / "experimentos" / "resultados" / "hipoteses.json"
SAIDA = RAIZ / "site" / "data" / "resultados.json"

TEXTO = {"data", "modelo", "metodo", "mlx_lm", "mlx", "python", "commit", "git_commit", "status", "fonte", "ferramenta", "tipo", "llamacpp_versao"}


def valor(coluna, bruto):
    bruto = (bruto or "").strip()
    if bruto == "":
        return None
    if coluna in TEXTO:
        return bruto
    try:
        v = float(bruto)
    except ValueError:
        return bruto
    if math.isnan(v) or math.isinf(v):
        return None
    return int(v) if v.is_integer() and coluna in {"bits", "grupo", "seq", "tokens_avaliados", "seed"} else v


def main():
    origem = CSV if CSV.exists() else PILOTO
    with origem.open(newline="", encoding="utf-8") as f:
        leitor = csv.DictReader(f)
        colunas = list(leitor.fieldnames or [])
        linhas = [{c: valor(c, r.get(c)) for c in colunas} for r in leitor]
    fonte = "matriz" if origem == CSV else "piloto"
    for l in linhas:
        l["fonte"], l["ferramenta"] = fonte, "mlx-lm"
    linhas = [l for l in linhas if not str(l.get("status") or "").startswith("falhou")]

    # Linha fp mais recente de cada modelo = referência da razão PPL_fp/PPL_q.
    ref = {}
    for l in linhas:
        if l.get("metodo") == "fp" and l.get("ppl_wikitext2"):
            ref[l.get("modelo")] = l["ppl_wikitext2"]
    for l in linhas:
        fp, q = ref.get(l.get("modelo")), l.get("ppl_wikitext2")
        l["razao_ppl"] = round(fp / q, 6) if fp and q else None
        # Perda relativa de qualidade (1 - razão), em %, útil nas dicas
        l["perda_pct"] = round((1 - l["razao_ppl"]) * 100, 4) if l["razao_ppl"] is not None else None

    # K-quants (llama.cpp): a PPL absoluta usa outro janelamento e NAO e comparavel a do MLX;
    # so a razao contra o F16 do proprio llama.cpp entra nos graficos.
    if KQUANT.exists():
        kq = list(csv.DictReader(KQUANT.open(newline="", encoding="utf-8")))
        f16 = {r["modelo"]: float(r["ppl"]) for r in kq if r["tipo"] == "F16"}
        for r in kq:
            if r["tipo"] == "F16" or r["modelo"] not in f16:
                continue
            ppl = float(r["ppl"])
            razao = round(f16[r["modelo"]] / ppl, 6)
            linhas.append({"modelo": r["modelo"], "metodo": "kquant", "tipo": r["tipo"], "bits": int(r["bits_nominais"]),
                           "bits_por_peso": float(r["bits_por_peso"]), "tamanho_gb": float(r["tamanho_gb"]),
                           "ppl_wikitext2": None, "ppl_llamacpp": ppl, "kld_media": float(r["kld_media"]),
                           "razao_ppl": razao, "perda_pct": round((1 - razao) * 100, 4), "data": r["data"],
                           "fonte": fonte, "ferramenta": "llama.cpp " + r["llamacpp_versao"]})

    hip = None
    if HIPOTESES.exists():
        try:
            hip = json.loads(HIPOTESES.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"aviso: hipoteses.json inválido ({e}); ignorado")

    saida = {
        "gerado_em": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fonte": fonte,
        "colunas": colunas + ["razao_ppl", "perda_pct", "fonte", "ferramenta"],
        "n_linhas": len(linhas),
        "modelos": sorted({l["modelo"] for l in linhas if l.get("modelo")}),
        "linhas": linhas,
        "hipoteses": hip,
    }
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(json.dumps(saida, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(linhas)} linhas -> {SAIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
