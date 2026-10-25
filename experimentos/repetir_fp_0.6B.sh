#!/bin/zsh
# Repete a avaliacao de qualidade do Qwen3-0.6B-fp depois que a etapa 4 terminar.
# Motivo: na 1a rodada levou 31962s (~8h53) contra ~25 min do 1.7B-fp, provavelmente por swap.
# Grava em um CSV separado para nao sobrescrever resultados.csv; so serve para comparar tempo e metricas.
#   caffeinate -i -s zsh experimentos/repetir_fp_0.6B.sh <pid da etapa 4>
set -u
cd "$(dirname "$0")/.."
PID=${1:?informe o pid do executar_etapa4.sh}
LOG=experimentos/logs/repetir_fp_0.6B.log
echo "== $(date) aguardando etapa 4 (pid $PID)" >> $LOG
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "== $(date) inicio; memoria livre: $(.venv/bin/python -c 'import psutil;print(round(psutil.virtual_memory().available/1e9,1))') GB" >> $LOG
INI=$(date +%s)
.venv/bin/python experimentos/rodar.py --modelo Qwen/Qwen3-0.6B --metodo fp --fase qualidade --refazer \
  --csv experimentos/resultados/repeticao-fp-0.6B.csv >> $LOG 2>&1 \
  && echo "[ok] Qwen3-0.6B-fp ($(( $(date +%s) - INI ))s; 1a rodada: 31962s)" >> $LOG \
  || echo "[FALHOU] Qwen3-0.6B-fp ($(( $(date +%s) - INI ))s)" >> $LOG
echo "== $(date) fim" >> $LOG
