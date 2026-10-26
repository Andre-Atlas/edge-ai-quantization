"""Prepara a rodada 2 da fase de velocidade (app fechado).

  python experimentos/preparar_rodada2_velocidade.py arquivar   # tabela so com as colunas de velocidade da rodada 1
  python experimentos/preparar_rodada2_velocidade.py zerar      # zera essas colunas em resultados.csv e move kquant-velocidade.csv
A rodada 1 (app Claude aberto) fica preservada em resultados/velocidade-rodada1-app-aberto.csv, kquant-velocidade-rodada1-app-aberto.csv
e em resultados/backups/. Usado por repetir_velocidade_app_fechado.sh.
"""
import csv
import os
import sys

R = "experimentos/resultados/"
VEL = ["prefill_tps", "prefill_tps_dp", "decode_tps", "decode_tps_dp", "pico_memoria_gb", "data_velocidade", "commit_velocidade"]
CHAVE = ["modelo", "metodo", "bits", "grupo", "seed"]

if sys.argv[1] == "arquivar":
    rows = [r for r in csv.DictReader(open(R + "resultados.csv")) if r.get("decode_tps")]
    with open(R + "velocidade-rodada1-app-aberto.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CHAVE + VEL, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[rodada2] rodada 1 arquivada: {len(rows)} linhas MLX")
else:
    rd = csv.DictReader(open(R + "resultados.csv"))
    campos, rows = rd.fieldnames, list(rd)
    for r in rows:
        for c in VEL + ["erro_velocidade"]:
            if c in r:
                r[c] = ""
    with open(R + "resultados.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(rows)
    os.replace(R + "kquant-velocidade.csv", R + "kquant-velocidade-rodada1-app-aberto.csv")
    print("[velocidade-r2] colunas de velocidade zeradas em resultados.csv; K-quants da rodada 1 movidos")
