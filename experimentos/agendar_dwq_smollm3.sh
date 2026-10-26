#!/bin/zsh
# Tentativa do DWQ do SmolLM3-3B a 4 bits (caso a parte; so o 4 bits, a celula central): depois da fila dos GPTQ restantes, sozinho
# na maquina. Roda o DWQ PADRAO do mlx-lm sob medir_pico_quant.py (trava: encerra o filho se a memoria disponivel cair abaixo de
# 1,5 GB), porque a estimativa (22,6 GB) passa da RAM e nao ha implementacao de menor memoria (o sublote foi abortado). Se gerar
# o modelo, move-o para modelos/ e roda a qualidade (rodar.py reaproveita o modelo existente) e registra impl/tempo. Se a trava
# abortar, nada e gravado no CSV; o resultado fica no log (limite inferior do pico). Job de uso unico no launchd.
set -u
cd "$(dirname "$0")/.."
LOG=experimentos/logs/etapa4.log
BASE=/private/tmp/claude-501/dwq_smollm3   # pasta SEM espacos (o shlex do medir_pico_quant separa por espaco)
ID=SmolLM3-3B-dwq-4b-g64-s0
echo "== $(date) [dwq-smollm3] agendado; espera o fim da fila dos GPTQ restantes" >> $LOG
while launchctl list 2>/dev/null | grep -q com.edgeai.gptq-restantes; do sleep 60; done
echo "== $(date) [dwq-smollm3] iniciando DWQ 4 bits do SmolLM3-3B (padrao, com trava), sozinho" >> $LOG
rm -rf $BASE; mkdir -p $BASE
.venv/bin/python experimentos/medir_pico_quant.py --modelo HuggingFaceTB/SmolLM3-3B --metodos dwq --bits 4 --piso-gb 1.5 \
  --saida $BASE/modelo > $BASE/medicao.log 2>&1
grep -E "^\[(footprint|ok|FALHOU|medindo)\]" $BASE/medicao.log | sed 's/^/[dwq-smollm3] /' >> $LOG
if [ -f $BASE/modelo/config.json ]; then
  rm -rf experimentos/modelos/$ID; mv $BASE/modelo experimentos/modelos/$ID
  .venv/bin/python experimentos/rodar.py --modelo HuggingFaceTB/SmolLM3-3B --metodo dwq --bits 4 --grupo 64 --seed 0 --fase qualidade \
    > experimentos/logs/$ID.qualidade.log 2>&1 && echo "[dwq-smollm3] [ok] qualidade" >> $LOG || echo "[dwq-smollm3] [FALHOU] qualidade" >> $LOG
  .venv/bin/python experimentos/conferir_nan.py >> $LOG 2>&1
else
  echo "[dwq-smollm3] sem modelo gerado (trava ou falha): nada gravado em resultados.csv" >> $LOG
fi
echo "== $(date) [dwq-smollm3] fim" >> $LOG
rm -f ~/Library/LaunchAgents/com.edgeai.dwq-smollm3.plist
launchctl bootout gui/$(id -u)/com.edgeai.dwq-smollm3 2>/dev/null
