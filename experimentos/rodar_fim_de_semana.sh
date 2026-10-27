#!/bin/zsh
# Extensao da protocolo-v3: roda a grade de matriz-v3.json SO na janela permitida pelo usuario (sexta 18h -> segunda, dia
# inteiro; nenhum job novo comeca depois de segunda 22h). Disparado toda sexta 18h pelo launchd (com.edgeai.v3-fds); retomavel:
# cada fim de semana continua de onde o anterior parou. Passos:
#  0) fora da janela ou outra instancia rodando -> sai; backup do resultados.csv;
#  1) validacao (so na 1a vez; protocolo-v3, secao 4): regressao do gptq_camada.py bf16 a 4 bits contra o GPTQ padrao do
#     Qwen3-0,6B (tensores identicos) e sanidade do GPTQ-fp32 a 3 bits (PPL finita, RTN3 > GPTQ3 > GPTQ4). Se falhar, o
#     gptq_fp32 sai da fila e o motivo fica no log;
#  2) matriz.py --config matriz-v3.json --ate <segunda 22h>, so com os modelos ja baixados (o Llama-3.2-1B depende de acesso);
#  3) conferir_nan.py; integridade das linhas que ja existiam (nenhuma coluna de qualidade/status mudou); resumo;
#  4) se nao restar nada da grade, remove o agendamento.
#   caffeinate -i -s zsh experimentos/rodar_fim_de_semana.sh
set -u
cd "$(dirname "$0")/.."
LOG=experimentos/logs/etapa4.log
PY=.venv/bin/python
LOCK=experimentos/logs/.v3_fds.lock
VALID=experimentos/logs/.v3_validacao

ATE=$($PY - <<'E'
import datetime as dt
h = dt.datetime.now()
# sexta = 4, sabado = 5, domingo = 6, segunda = 0
dias = {4: 3, 5: 2, 6: 1, 0: 0}.get(h.weekday())
if dias is None or (h.weekday() == 4 and h.hour < 18):
    print("")
else:
    print((h + dt.timedelta(days=dias)).strftime("%Y-%m-%d") + " 22:00")
E
)
if [[ -z $ATE ]]; then echo "== $(date) [v3-fds] fora da janela (sexta 18h a segunda): nada a fazer" >> $LOG; exit 0; fi
if [[ "$(date '+%Y-%m-%d %H:%M')" > "$ATE" ]]; then echo "== $(date) [v3-fds] depois de segunda 22h: nada a iniciar" >> $LOG; exit 0; fi
if [[ -e $LOCK ]] && kill -0 "$(cat $LOCK)" 2>/dev/null; then echo "== $(date) [v3-fds] ja em execucao (pid $(cat $LOCK))" >> $LOG; exit 0; fi
echo $$ > $LOCK
BKP=experimentos/resultados/backups/resultados.csv.antes-v3-fds-$(date +%Y%m%d-%H%M)
cp experimentos/resultados/resultados.csv $BKP
echo "== $(date) [v3-fds] inicio; janela ate $ATE; backup $BKP; memoria livre $($PY -c 'import psutil;print(round(psutil.virtual_memory().available/1e9,1))') GB" >> $LOG

METODOS=()
# ---- 1) validacao (uma vez) ----
if [[ ! -e $VALID ]]; then
  echo "== $(date) [v3-validacao] regressao: gptq_camada.py bf16 4 bits no Qwen3-0,6B contra o GPTQ padrao" >> $LOG
  REG=/private/tmp/claude-501/v3_regressao
  rm -rf $REG
  $PY experimentos/gptq_camada.py --model Qwen/Qwen3-0.6B --mlx-path $REG --bits 4 --group-size 64 --seed 0 --hessiana bf16 \
    > experimentos/logs/v3_regressao.log 2>&1
  $PY experimentos/comparar_gptq.py $REG experimentos/modelos/Qwen3-0.6B-gptq-4b-g64-s0 2>&1 | grep -E "^(tensores|identicos)" \
    | sed 's/^/[v3-validacao] /' >> $LOG
  reg_ok=$($PY experimentos/comparar_gptq.py $REG experimentos/modelos/Qwen3-0.6B-gptq-4b-g64-s0 2>/dev/null | grep -c "diferentes: 0$")
  echo "== $(date) [v3-validacao] GPTQ-fp32 4 e 3 bits no Qwen3-0,6B" >> $LOG
  $PY experimentos/matriz.py --fase qualidade --config matriz-v3.json --somente Qwen/Qwen3-0.6B --metodos gptq_fp32 --ate "$ATE" >> $LOG 2>&1
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
  if [[ $reg_ok == 1 && $san_ok == 1 ]]; then
    echo "[v3-validacao] OK: regressao identica e GPTQ-fp32 3 bits sao (RTN3 > GPTQ3 > GPTQ4)" >> $LOG
    echo ok > $VALID
  else
    echo "[v3-validacao] FALHOU (regressao=$reg_ok, sanidade=$san_ok): gptq_fp32 fora da fila ate decisao registrada" >> $LOG
    echo falhou > $VALID
  fi
fi
[[ $(cat $VALID) == falhou ]] && METODOS=(--metodos fp,rtn,awq,dwq)

# ---- 2) fila da v3, so com modelos ja baixados ----
SOMENTE=($($PY - <<'E'
import json
from huggingface_hub import try_to_load_from_cache
cfg = json.load(open("experimentos/matriz-v3.json"))
ids = {m for g in cfg["grades"] for m in g["modelos"]}
print(" ".join(sorted(m for m in ids if isinstance(try_to_load_from_cache(m, "config.json"), str))))
E
))
echo "== $(date) [v3-fds] fila (modelos baixados: ${SOMENTE[*]}) ${METODOS[*]}" >> $LOG
if (( ${#SOMENTE} )); then   # lista vazia faria o --somente valer "todos" (inclusive modelos nao baixados)
  $PY experimentos/matriz.py --fase qualidade --config matriz-v3.json --ate "$ATE" --somente $SOMENTE $METODOS >> $LOG 2>&1
else
  echo "[v3-fds] nenhum modelo baixado: fila nao iniciada" >> $LOG
fi

# ---- 3) NaN, integridade e resumo ----
$PY experimentos/conferir_nan.py >> $LOG 2>&1
BKP=$BKP $PY - >> $LOG 2>&1 <<'E'
import csv, os
CH = ["modelo", "metodo", "bits", "grupo", "seed"]
k = lambda r: tuple(r.get(c, "") for c in CH)
antes = {k(r): r for r in csv.DictReader(open(os.environ["BKP"]))}
depois = {k(r): r for r in csv.DictReader(open("experimentos/resultados/resultados.csv"))}
fixas = ["ppl_wikitext2", "acc_media", "tamanho_gb", "data", "commit", "impl"]
difs = [f"{c[0].split('/')[-1]} {c[1]} {c[2]}b: {f}" for c, a in antes.items() if a.get("acc_media")
        for f in fixas if depois.get(c, {}).get(f, "") != a.get(f, "")]
difs += [f"{c[0].split('/')[-1]} {c[1]} {c[2]}b: status" for c, a in antes.items()
         if a.get("status", "").startswith("falhou:invalido") and depois.get(c, {}).get("status") != a["status"]]
novas = [c for c in depois if c not in antes or not antes[c].get("acc_media")]
ok = [c for c in novas if depois[c].get("acc_media") and not depois[c]["status"].startswith("falhou")]
print(f"[v3-integridade] {'OK' if not difs else 'FALHOU'}: {len(difs)} diferencas nas linhas que ja existiam; "
      f"linhas novas com qualidade: {len(novas)} ({len(ok)} validas)")
for d in difs[:20]:
    print("[v3-integridade]   ", d)
E
RESTAM=$($PY experimentos/matriz.py --fase qualidade --config matriz-v3.json --simular 2>/dev/null | head -1 | awk '{print $1}')
echo "== $(date) [v3-fds] fim desta janela; restam $RESTAM job(s) da grade v3" >> $LOG
rm -f $LOCK
if [[ $RESTAM == 0 ]]; then
  echo "== $(date) [v3-fds] grade v3 completa: agendamento removido" >> $LOG
  rm -f ~/Library/LaunchAgents/com.edgeai.v3-fds.plist
  launchctl bootout gui/$(id -u)/com.edgeai.v3-fds 2>/dev/null
fi
