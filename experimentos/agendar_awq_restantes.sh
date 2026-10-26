#!/bin/zsh
# AWQ (awq_liberar.py) do Llama-3.2-3B e do Qwen3-4B nas larguras que faltam, em serie e sozinhos na maquina: 3, 2 e 8 bits
# (3 e 2 bits primeiro: alimentam a curva de qualidade x bits e a analise secundaria da H3; 8 bits e o menos informativo).
# Os de 4 bits ja estao validos (25/10). Ao fim de cada job confere a perplexidade ([conferencia]) para detectar NaN.
# Um job que falhe nao interrompe os seguintes.
#   caffeinate -i -s zsh experimentos/agendar_awq_restantes.sh
set -u
cd "$(dirname "$0")/.."
LOG=experimentos/logs/etapa4.log
echo "== $(date) iniciando AWQ 3, 2 e 8 bits (awq_liberar.py) no Llama-3.2-3B e no Qwen3-4B, em serie" >> $LOG
for par in "meta-llama/Llama-3.2-3B:Llama-3.2-3B" "Qwen/Qwen3-4B:Qwen3-4B"; do
  M=${par%%:*}
  for B in 3 2 8; do
    ID=${par#*:}-awq-${B}b-g64-s0
    echo "[inicio] $ID (awq_liberar)" >> $LOG
    INI=$(date +%s)
    .venv/bin/python experimentos/rodar.py --modelo $M --metodo awq --bits $B --grupo 64 --seed 0 --fase qualidade --awq-liberar \
      > experimentos/logs/$ID.qualidade.log 2>&1 \
      && echo "[ok] $ID ($(( $(date +%s) - INI ))s)" >> $LOG || echo "[FALHOU] $ID ($(( $(date +%s) - INI ))s)" >> $LOG
    grep -h "pico_quant" experimentos/logs/$ID.qualidade.log | sed 's/^/[conferencia] /' >> $LOG
    MODELO=$M BITS=$B .venv/bin/python - >> $LOG 2>&1 <<'E'
import csv, math, os
m, b = os.environ["MODELO"], os.environ["BITS"]
for r in csv.DictReader(open("experimentos/resultados/resultados.csv")):
    if r["modelo"] == m and r["metodo"] == "awq" and r["bits"] == b and r["seed"] == "0":
        ppl = r.get("ppl_wikitext2", "")
        ruim = ppl == "" or math.isnan(float(ppl))
        print(f"[conferencia] {m} awq {b}b: PPL={ppl or 'vazio'} acc_media={r.get('acc_media') or 'vazio'} status={r['status']} "
              + ("-> ATENCAO: PPL NaN ou ausente" if ruim else "-> PPL valida"))
E
  done
done
echo "== $(date) fim do AWQ restante (Llama e Qwen3-4B; 3, 2 e 8 bits)" >> $LOG
