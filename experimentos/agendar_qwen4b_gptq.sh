#!/bin/zsh
# Depois do GPTQ do Llama a 4 bits, roda o GPTQ do Qwen3-4B a 4 bits com o gptq_camada.py (identico ao mlx-lm, Hessiana bf16,
# ~3-4x menos memoria). Sozinho na maquina. 8 e 2 bits ficam para depois, conforme o resultado de 4 bits.
# Ao fim confere a perplexidade: o GPTQ do SmolLM3 saiu NaN, e o 4B pode ter o mesmo defeito (arredondamento em bf16).
#   caffeinate -i -s zsh experimentos/agendar_qwen4b_gptq.sh <pid do agendar_llama_gptq.sh>
set -u
cd "$(dirname "$0")/.."
PID=${1:?informe o pid do agendar_llama_gptq.sh}
LOG=experimentos/logs/etapa4.log
ID=Qwen3-4B-gptq-4b-g64-s0
echo "== $(date) agendado: GPTQ do Qwen3-4B a 4 bits (camada a camada) apos o Llama (pid $PID)" >> $LOG
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "== $(date) iniciando GPTQ do Qwen3-4B a 4 bits (gptq_camada.py), sozinho" >> $LOG
echo "[inicio] $ID (camada a camada, ~17 GB)" >> $LOG
INI=$(date +%s)
.venv/bin/python experimentos/rodar.py --modelo Qwen/Qwen3-4B --metodo gptq --bits 4 --grupo 64 --seed 0 --fase qualidade \
  --gptq-camada > experimentos/logs/$ID.qualidade.log 2>&1 \
  && echo "[ok] $ID ($(( $(date +%s) - INI ))s)" >> $LOG \
  || echo "[FALHOU] $ID ($(( $(date +%s) - INI ))s)" >> $LOG
.venv/bin/python - >> $LOG 2>&1 <<'E'
import csv, math
for r in csv.DictReader(open("experimentos/resultados/resultados.csv")):
    if r["modelo"] == "Qwen/Qwen3-4B" and r["metodo"] == "gptq" and r["bits"] == "4":
        ppl = r.get("ppl_wikitext2", "")
        ruim = ppl == "" or math.isnan(float(ppl))
        print(f"[conferencia] {r['modelo']} gptq 4b: PPL={ppl or 'vazio'} acc_media={r.get('acc_media') or 'vazio'} status={r['status']} "
              + ("-> ATENCAO: PPL NaN ou ausente; marcar como invalido como no SmolLM3" if ruim else "-> PPL valida"))
E
echo "== $(date) fim do GPTQ do Qwen3-4B a 4 bits" >> $LOG
