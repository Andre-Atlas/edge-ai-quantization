#!/bin/zsh
# Continua a lista depois do GPTQ do Llama a 4 bits: P1 replicas de AWQ (sementes 1 e 2 em Qwen3-0,6B e 1,7B) e
# P2 K-quants (llama.cpp, em serie). NAO inicia a fase de velocidade (H4): ela exige a maquina totalmente ociosa e
# deve ser lancada a mao, com o usuario longe do Mac. Tambem nao roda o GPTQ do Llama a 8 e 2 bits (opcional, depois).
#   caffeinate -i -s zsh experimentos/continuar_lista.sh <pid do agendar_llama_gptq.sh>
set -u
cd "$(dirname "$0")/.."
PID=${1:?informe o pid do agendar_llama_gptq.sh}
LOG=experimentos/logs/etapa4.log
echo "== $(date) agendado: replicas de AWQ e K-quants apos o GPTQ do Llama (pid $PID)" >> $LOG
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "== $(date) continuando a lista: replicas de AWQ (qualidade)" >> $LOG
.venv/bin/python experimentos/matriz.py --fase qualidade --metodos awq --somente Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B --workers 1 >> $LOG 2>&1
echo "== $(date) K-quants (em serie)" >> $LOG
export PATH="$HOME/.local/bin:$PATH"
for m in Qwen/Qwen3-1.7B HuggingFaceTB/SmolLM3-3B meta-llama/Llama-3.2-3B Qwen/Qwen3-4B; do
  if grep -q "^$m,kquant,Q2_K" experimentos/resultados/kquant.csv 2>/dev/null; then echo "kquant $m ja feito" >> $LOG; continue; fi
  echo "[inicio] kquant $m" >> $LOG
  .venv/bin/python experimentos/kquant.py --modelo "$m" --tipos Q8_0,Q4_K_M,Q3_K_M,Q2_K > "experimentos/logs/kquant-${m//\//_}.log" 2>&1 \
    && echo "[ok] kquant $m" >> $LOG || echo "[FALHOU] kquant $m" >> $LOG
done
echo "== $(date) fim da lista automatica; velocidade NAO iniciada (precisa da maquina ociosa)" >> $LOG
