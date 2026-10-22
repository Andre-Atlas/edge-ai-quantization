#!/bin/zsh
# Disparado pelo launchd (~/Library/LaunchAgents/com.edgeai.etapa4.plist) a 00:00.
# Roda uma unica vez: o marcador impede repeticao nas noites seguintes.
cd "$(dirname "$0")/.."
MARCA=experimentos/logs/.etapa4_iniciada
[[ -e $MARCA ]] && { echo "$(date) ja iniciada antes; nada a fazer"; exit 0; }
date > $MARCA
echo "== $(date) inicio agendado; memoria livre: $(.venv/bin/python -c 'import psutil;print(round(psutil.virtual_memory().available/1e9,1))') GB"
# -i: sem repouso por inatividade; -s: sem repouso na tomada (a tela pode apagar)
exec /usr/bin/caffeinate -i -s /bin/zsh experimentos/executar_etapa4.sh 1
