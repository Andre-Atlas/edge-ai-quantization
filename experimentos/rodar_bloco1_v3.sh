#!/bin/zsh
# Executa o Bloco 1 da protocolo-v3 (decisao D27): GPTQ-fp32 (4 e 3 bits) apenas nos 5 modelos da v2
# (Qwen3-0.6B, Qwen3-1.7B, SmolLM3-3B, Llama-3.2-3B e Qwen3-4B) para comprovar a eliminacao do NaN
# pela Hessiana em float32 sem atrasar o cronograma do artigo.
#
# Regras de seguranca:
#  1) Exige tomada (AC Power) para evitar o incidente de repouso de 30/10 (Fase 24);
#  2) Faz backup de resultados.csv antes de iniciar;
#  3) Verifica sanidade de 3 bits no Qwen3-0.6B (PPL_RTN3 > PPL_GPTQ_fp32_3 > PPL_GPTQ_fp32_4);
#  4) Ao final, roda conferir_nan.py, checa integridade das linhas anteriores e atualiza analise/site.
#
# Uso:
#   caffeinate -i -s zsh experimentos/rodar_bloco1_v3.sh
set -u
cd "$(dirname "$0")/.."
LOG=experimentos/logs/etapa4.log
PY=.venv/bin/python
LOCK=experimentos/logs/.v3_bloco1.lock

if [[ -e $LOCK ]] && kill -0 "$(cat $LOCK)" 2>/dev/null; then
  echo "== $(date) [v3-bloco1] ja em execucao (pid $(cat $LOCK))" >> $LOG
  exit 0
fi

if ! pmset -g batt | head -1 | grep -q "AC Power"; then
  echo "== $(date) [v3-bloco1] Mac na bateria: nada iniciado (ligue na tomada e rode de novo)" | tee -a $LOG
  exit 1
fi

echo $$ > $LOCK
trap 'rm -f $LOCK' EXIT

BKP=experimentos/resultados/backups/resultados.csv.antes-v3-bloco1-$(date +%Y%m%d-%H%M)
cp experimentos/resultados/resultados.csv "$BKP"
echo "== $(date) [v3-bloco1] inicio; backup $BKP; memoria livre $($PY -c 'import psutil;print(round(psutil.virtual_memory().available/1e9,1))') GB" >> $LOG

# 1) Concluir/verificar Qwen3-0.6B (4 e 3 bits) e validar criterio da secao 4.2 do protocolo-v3
$PY experimentos/matriz.py --fase qualidade --config matriz-v3.json --somente Qwen/Qwen3-0.6B --metodos gptq_fp32 >> $LOG 2>&1

san_ok=$($PY - <<'E'
import csv, math
r = {(x["metodo"], x["bits"]): x for x in csv.DictReader(open("experimentos/resultados/resultados.csv"))
     if x["modelo"] == "Qwen/Qwen3-0.6B" and x["seed"] == "0" and not x["status"].startswith("falhou")}
try:
    rtn3, g3, g4 = (float(r[k]["ppl_wikitext2"]) for k in [("rtn", "3"), ("gptq_fp32", "3"), ("gptq_fp32", "4")])
    print(1 if all(map(math.isfinite, (rtn3, g3, g4))) and rtn3 > g3 > g4 else 0)
except (KeyError, ValueError):
    print(0)
E
)

if [[ $san_ok != 1 ]]; then
  echo "== $(date) [v3-bloco1] FALHOU sanidade 3 bits no Qwen3-0.6B: interrompendo fila" | tee -a $LOG
  exit 1
fi
echo "== $(date) [v3-bloco1] sanidade 3 bits OK (RTN3 > GPTQ_fp32_3 > GPTQ_fp32_4)" | tee -a $LOG

# 2) Executar GPTQ-fp32 (4 e 3 bits) nos demais modelos da v2
MODELOS_V2=(
  "Qwen/Qwen3-0.6B"
  "Qwen/Qwen3-1.7B"
  "HuggingFaceTB/SmolLM3-3B"
  "meta-llama/Llama-3.2-3B"
  "Qwen/Qwen3-4B"
)

$PY experimentos/matriz.py --fase qualidade --config matriz-v3.json --somente "${MODELOS_V2[@]}" --metodos gptq_fp32 >> $LOG 2>&1

# 3) Conferir NaN, integridade e atualizar analises, site e tabelas do artigo
$PY experimentos/conferir_nan.py >> $LOG 2>&1
$PY experimentos/analise.py >> $LOG 2>&1
$PY experimentos/analise_v3.py >> $LOG 2>&1
$PY experimentos/exportar_site.py >> $LOG 2>&1
$PY experimentos/gerar_tabelas_artigo.py >> $LOG 2>&1
echo "== $(date) [v3-bloco1] concluido" | tee -a $LOG
