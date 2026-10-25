#!/bin/zsh
# Vigia a fila da etapa 4 e SAI (codigo 0) imprimindo "STALL: ..." quando ela fica parada por mais de LIMITE segundos.
# Quem roda este script em segundo plano (o Claude Code) e acordado quando ele termina e envia o e-mail de aviso.
#   zsh experimentos/vigiar_espera.sh [limite_em_segundos=3600]
# Variaveis opcionais: LOG (padrao experimentos/logs/etapa4.log).
# Parada = (A) o fim do log e uma sequencia de "[aguardando memoria]" (um aviso a cada ~60 s) que dura >= LIMITE; ou
#          (B) ha fila/agendamento vivo, nenhum job rodando (rodar.py, mlx_lm, avaliar.py, kquant.py, llama.cpp)
#          e o log nao muda ha >= LIMITE.
set -u
cd "$(dirname "$0")/.."
LIMITE=${1:-3600}
LOG=${LOG:-experimentos/logs/etapa4.log}
ini_espera=0
while true; do
  agora=$(date +%s)
  # (A) avisos de espera consecutivos no fim do log: cada um ~60 s; soma tambem o tempo desde que este vigia viu a espera comecar
  n=$(tail -n 400 "$LOG" 2>/dev/null | awk '/^\[aguardando memoria\]/{c++; next} {c=0} END{print c+0}')
  if (( n > 0 )); then
    (( ini_espera == 0 )) && ini_espera=$agora
    esp=$(( (n - 1) * 60 )); (( agora - ini_espera > esp )) && esp=$(( agora - ini_espera ))
    if (( esp >= LIMITE )); then
      echo "STALL: fila esperando memoria ha ~$(( esp / 60 )) min ($(date))"
      echo "ultima linha: $(tail -n 1 "$LOG" | cut -c1-160)"
      echo "memoria: $(.venv/bin/python -c 'import psutil;v=psutil.virtual_memory();print(round(v.available/1e9,1),"GB disponiveis de",round(v.total/1e9,1))' 2>/dev/null)"
      exit 0
    fi
  else
    ini_espera=0
    # (B) fila viva, sem job e sem escrita no log
    if pgrep -f "matriz.py|agendar_llama_gptq|continuar_lista" > /dev/null && \
       ! pgrep -f "rodar.py|mlx_lm|avaliar.py|kquant.py|llama-quantize|llama-perplexity" > /dev/null; then
      mod=$(stat -f %m "$LOG" 2>/dev/null || echo "$agora")
      if (( agora - mod >= LIMITE )); then
        echo "STALL: fila viva, nenhum job rodando e log parado ha ~$(( (agora - mod) / 60 )) min ($(date))"
        echo "ultima linha: $(tail -n 1 "$LOG" | cut -c1-160)"
        exit 0
      fi
    fi
  fi
  sleep 60
done
