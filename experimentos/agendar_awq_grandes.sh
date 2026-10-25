#!/bin/zsh
# AWQ a 4 bits do Llama-3.2-3B e do Qwen3-4B pelo awq_liberar.py (identico ao mlx-lm; memoria liberada por bloco), um depois do
# outro, sozinhos na maquina. Valida a previsao de memoria ([pico_quant]) e confere a perplexidade ao fim de cada um.
# 8, 3 e 2 bits ficam para depois, conforme o resultado de 4 bits.
#   caffeinate -i -s zsh experimentos/agendar_awq_grandes.sh
set -u
cd "$(dirname "$0")/.."
LOG=experimentos/logs/etapa4.log
echo "== $(date) iniciando AWQ 4 bits (awq_liberar.py) no Llama-3.2-3B e no Qwen3-4B, em serie" >> $LOG
for par in "meta-llama/Llama-3.2-3B:Llama-3.2-3B" "Qwen/Qwen3-4B:Qwen3-4B"; do
  M=${par%%:*}; ID=${par#*:}-awq-4b-g64-s0
  echo "[inicio] $ID (awq_liberar, ~13-15 GB previstos)" >> $LOG
  INI=$(date +%s)
  .venv/bin/python experimentos/rodar.py --modelo $M --metodo awq --bits 4 --grupo 64 --seed 0 --fase qualidade --awq-liberar \
    > experimentos/logs/$ID.qualidade.log 2>&1 \
    && echo "[ok] $ID ($(( $(date +%s) - INI ))s)" >> $LOG || echo "[FALHOU] $ID ($(( $(date +%s) - INI ))s)" >> $LOG
  grep -h "pico_quant" experimentos/logs/$ID.qualidade.log | sed 's/^/[conferencia] /' >> $LOG
  MODELO=$M .venv/bin/python - >> $LOG 2>&1 <<'E'
import csv, math, os
m = os.environ["MODELO"]
for r in csv.DictReader(open("experimentos/resultados/resultados.csv")):
    if r["modelo"] == m and r["metodo"] == "awq" and r["bits"] == "4" and r["seed"] == "0":
        ppl = r.get("ppl_wikitext2", "")
        ruim = ppl == "" or math.isnan(float(ppl))
        print(f"[conferencia] {m} awq 4b: PPL={ppl or 'vazio'} acc_media={r.get('acc_media') or 'vazio'} status={r['status']} "
              + ("-> ATENCAO: PPL NaN ou ausente" if ruim else "-> PPL valida"))
E
done
echo "== $(date) fim do AWQ dos modelos grandes (4 bits)" >> $LOG
