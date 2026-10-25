#!/bin/zsh
# Teste completo do DWQ com alvos do professor pre-calculados (--target-dir) no Qwen3-0,6B, 4 bits, sozinho na maquina:
#  (1) mede o pico de memoria das duas fases (alvos e treino) e o footprint; (2) compara a qualidade com o DWQ padrao do mesmo
#  modelo (razao Q_variante/Q_padrao com IC95% pareado). O 1o teste (11:57) foi abortado pela trava aos 12 s porque outros
#  jobs ocupavam a memoria; os alvos usam top-1024 logits, entao o resultado NAO e identico ao padrao.
#   caffeinate -i -s zsh experimentos/testar_dwq_alvos.sh <pid do agendar_qwen4b_gptq.sh>
set -u
cd "$(dirname "$0")/.."
PID=${1:?informe o pid do agendar_qwen4b_gptq.sh}
LOG=experimentos/logs/etapa4.log
# pasta SEM espacos (o shlex do medir_pico_quant separa argumentos por espaco); fora do projeto, apagavel
BASE=/private/tmp/claude-501/dwq_alvos_teste
echo "== $(date) agendado: teste DWQ com alvos pre-calculados (0,6B) apos o GPTQ do Qwen3-4B (pid $PID)" >> $LOG
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "== $(date) iniciando teste DWQ com alvos pre-calculados (0,6B), sozinho" >> $LOG
rm -rf $BASE; mkdir -p $BASE
INI=$(date +%s)
.venv/bin/python experimentos/medir_pico_quant.py --modelo Qwen/Qwen3-0.6B --metodos dwq --bits 4 --piso-gb 1.5 \
  --saida $BASE/modelo --extra "--target-dir $BASE/alvos" > $BASE/medicao.log 2>&1
grep -E "^\[(footprint|ok|FALHOU)\]" $BASE/medicao.log | sed 's/^/[dwq_alvos] /' >> $LOG
echo "[dwq_alvos] duracao $(( $(date +%s) - INI )) s; alvos em disco: $(du -sh $BASE/alvos 2>/dev/null | cut -f1)" >> $LOG
if [ -f $BASE/modelo/config.json ]; then
  (cd experimentos && ../.venv/bin/python comparar_gptq.py $BASE/modelo modelos/Qwen3-0.6B-dwq-4b-g64-s0 \
     --nll resultados/nll/Qwen3-0.6B-dwq-4b-g64-s0.npy 2>&1 | grep -E "^(tensores|identicos|NLL|razao)" | sed 's/^/[dwq_alvos] /' >> ../$LOG)
else
  echo "[dwq_alvos] sem modelo gerado (abortado ou falhou): sem comparacao de qualidade" >> $LOG
fi
echo "== $(date) fim do teste DWQ com alvos (pasta $BASE; apagar quando nao for mais preciso)" >> $LOG
