#!/bin/zsh
# Dois diagnosticos de memoria/viabilidade, sozinhos na maquina, depois da cadeia principal (K-quants) e do teste da Hessiana:
#  (1) DWQ com alvos do professor em sub-lotes (dwq_sublote.py) no Qwen3-0,6B, 4 bits: pico de memoria + qualidade contra o
#      DWQ padrao do mesmo modelo (razao Q com IC95% pareado). Os alvos guardam so os 1024 maiores logits: NAO e identico ao padrao.
#  (2) Perfil de memoria do AWQ (perfil_awq.py) no Qwen3-0,6B, 4 bits, com os parametros do estudo: onde esta o pico e se a
#      memoria se acumula de um bloco para outro.
# Resultados no etapa4.log com os prefixos [diag_dwq] e [diag_awq]. Nao grava nada em resultados.csv.
#   caffeinate -i -s zsh experimentos/rodar_diagnosticos.sh <pid do continuar_lista.sh>
set -u
cd "$(dirname "$0")/.."
PID=${1:?informe o pid do continuar_lista.sh}
LOG=experimentos/logs/etapa4.log
BASE=/private/tmp/claude-501/diag_dwq_awq        # sem espacos no caminho (o argparse/shlex do dwq separa por espaco)
echo "== $(date) agendado: diagnosticos DWQ (alvos em sub-lotes) e perfil de memoria do AWQ, no Qwen3-0,6B (apos pid $PID)" >> $LOG
while kill -0 "$PID" 2>/dev/null || pgrep -f "testar_hessiana.py" > /dev/null; do sleep 60; done
rm -rf $BASE; mkdir -p $BASE

echo "== $(date) [diag_dwq] iniciando, sozinho" >> $LOG
INI=$(date +%s)
/usr/bin/time -l .venv/bin/python experimentos/dwq_sublote.py --piso-gb 1.5 --model Qwen/Qwen3-0.6B --mlx-path $BASE/dwq_modelo \
  --bits 4 --group-size 64 --seed 0 --grad-checkpoint --target-dir $BASE/dwq_alvos > $BASE/dwq.log 2>&1
tr '\r' '\n' < $BASE/dwq.log | grep -E "peak memory footprint|\[dwq_sublote\]|ABORT|Validation|peak_memory_gb|Traceback|Error" | tail -8 | sed 's/^/[diag_dwq] /' >> $LOG
echo "[diag_dwq] duracao $(( $(date +%s) - INI )) s; alvos em disco: $(du -sh $BASE/dwq_alvos 2>/dev/null | cut -f1)" >> $LOG
if [ -f $BASE/dwq_modelo/config.json ]; then
  (cd experimentos && ../.venv/bin/python comparar_gptq.py $BASE/dwq_modelo modelos/Qwen3-0.6B-dwq-4b-g64-s0 \
     --nll resultados/nll/Qwen3-0.6B-dwq-4b-g64-s0.npy 2>&1 | grep -E "^(NLL|razao)" | sed 's/^/[diag_dwq] /' >> ../$LOG)
else
  echo "[diag_dwq] sem modelo gerado (abortado ou falhou): sem comparacao de qualidade" >> $LOG
fi

echo "== $(date) [diag_awq] iniciando, sozinho" >> $LOG
INI=$(date +%s)
/usr/bin/time -l .venv/bin/python experimentos/perfil_awq.py --piso-gb 1.5 --model Qwen/Qwen3-0.6B --mlx-path $BASE/awq_modelo \
  --bits 4 --group-size 64 --seed 0 > $BASE/awq.log 2>&1
grep -E "peak memory footprint" $BASE/awq.log | sed 's/^/[diag_awq] /' >> $LOG
grep -E "awq_perfil\] (FIM|maior)" $BASE/awq.log | sed 's/^/[diag_awq] /' >> $LOG
# tendencia entre blocos: memoria ativa apos cada clip_block (ultimo passo do bloco) - primeiros, do meio e ultimos
grep "clip_block" $BASE/awq.log | awk 'NR<=3 || (NR>=13 && NR<=15) || NR>=26' | sed 's/^/[diag_awq] tendencia: /' >> $LOG
echo "[diag_awq] duracao $(( $(date +%s) - INI )) s; log completo: $BASE/awq.log" >> $LOG
echo "== $(date) fim dos diagnosticos (pasta $BASE; apagar quando nao for mais preciso)" >> $LOG
