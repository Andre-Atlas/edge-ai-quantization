#!/bin/zsh
# Espera a fila de RTN terminar e roda SO o GPTQ do Llama-3.2-3B a 4 bits (o que a H1 precisa), sozinho na maquina.
# GPTQ usa ~3,6x o tamanho bf16 (medido no Qwen3-1,7B): ~22 GB neste modelo, com swap; por isso nada roda junto.
# 8 e 2 bits ficam para depois, se sobrar tempo (nao agendados aqui).
#   caffeinate -i -s zsh experimentos/agendar_llama_gptq.sh <pid da fila de RTN>
set -u
cd "$(dirname "$0")/.."
PID=${1:?informe o pid da fila de RTN (matriz.py --metodos rtn)}
LOG=experimentos/logs/etapa4.log
ID=Llama-3.2-3B-gptq-4b-g64-s0
echo "== $(date) agendado: GPTQ do Llama a 4 bits apos a fila de RTN (pid $PID)" >> $LOG
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "== $(date) fila de RTN terminou; iniciando GPTQ do Llama a 4 bits, sozinho" >> $LOG
echo "[inicio] $ID (~23 GB, sozinho)" >> $LOG
INI=$(date +%s)
.venv/bin/python experimentos/rodar.py --modelo meta-llama/Llama-3.2-3B --metodo gptq --bits 4 --grupo 64 --seed 0 \
  --fase qualidade > experimentos/logs/$ID.qualidade.log 2>&1 \
  && echo "[ok] $ID ($(( $(date +%s) - INI ))s)" >> $LOG \
  || echo "[FALHOU] $ID ($(( $(date +%s) - INI ))s)" >> $LOG
echo "== $(date) fim do GPTQ do Llama a 4 bits; nada mais agendado" >> $LOG
