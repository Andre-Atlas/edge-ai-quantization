#!/bin/zsh
# Repete o GPTQ de 8 bits do Llama-3.2-3B, que falhou em 26/10 (processo encerrado de forma anormal apos as 28 camadas, sem erro
# no log; suspeita de pressao de memoria, nao confirmada). Espera o fim da fila dos GPTQ restantes, roda sozinho e confere NaN.
# Job de uso unico no launchd. Se falhar de novo, a linha fica falhou:qualidade e o log guarda o motivo.
set -u
cd "$(dirname "$0")/.."
LOG=experimentos/logs/etapa4.log
echo "== $(date) [llama-gptq8] agendado; espera o fim da fila dos GPTQ restantes" >> $LOG
while launchctl list 2>/dev/null | grep -q com.edgeai.gptq-restantes; do sleep 60; done
echo "== $(date) [llama-gptq8] repetindo GPTQ 8 bits do Llama-3.2-3B, sozinho" >> $LOG
.venv/bin/python experimentos/matriz.py --fase qualidade --metodos gptq --somente meta-llama/Llama-3.2-3B --workers 1 >> $LOG 2>&1
.venv/bin/python experimentos/conferir_nan.py >> $LOG 2>&1
echo "== $(date) [llama-gptq8] fim" >> $LOG
rm -f ~/Library/LaunchAgents/com.edgeai.llama-gptq8.plist
launchctl bootout gui/$(id -u)/com.edgeai.llama-gptq8 2>/dev/null
