"""Marca como invalidos (status falhou:invalido-nan) os jobs executados cuja perplexidade deu NaN. Faz backup antes de alterar.

  python experimentos/conferir_nan.py
Criterio igual ao usado a mao em 25/10 (protocolo-v2, secao 4): job executado e ppl_wikitext2 = nan -> invalido, nao "pendente".
Os valores medidos (acc_*, tamanho, tempo) sao preservados; so o status muda.
"""
import csv
import math
import shutil
import time

CSV = "experimentos/resultados/resultados.csv"
rd = csv.DictReader(open(CSV))
campos, rows = rd.fieldnames, list(rd)
mudou = []
for r in rows:
    p = r.get("ppl_wikitext2", "")
    if p and math.isnan(float(p)) and not r.get("status", "").startswith("falhou"):
        r["status"] = "falhou:invalido-nan"
        mudou.append(f"{r['modelo']} {r['metodo']} {r['bits']}b")
if mudou:
    shutil.copy(CSV, f"experimentos/resultados/backups/resultados.csv.antes-de-conferir-nan-{time.strftime('%Y%m%d-%H%M')}")
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(rows)
print(f"[conferir_nan] marcados como invalidos: {len(mudou)} {mudou}")
