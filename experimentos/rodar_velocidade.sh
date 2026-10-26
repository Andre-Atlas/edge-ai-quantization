#!/bin/zsh
# Fase de velocidade (H4) da etapa 4, com a maquina ociosa: 59 jobs MLX (qualidade valida) + K-quants (llama-bench).
#  1) backup do resultados.csv; 2) espera ESPERA_S para o usuario fechar os apps e sair; 3) pre-voo: espera ate nao haver
#  job pesado, memoria livre >= 10 GB e carga baixa por 2 min (registra no log as condicoes: apps abertos, memoria, bateria);
#  4) matriz.py --fase velocidade; 5) kquant_velocidade.py; 6) checagem de integridade das colunas de qualidade contra o
#  backup. Tudo no experimentos/logs/etapa4.log com o prefixo [velocidade].
#   caffeinate -i -s zsh experimentos/rodar_velocidade.sh [espera_em_segundos=300]
set -u
cd "$(dirname "$0")/.."
LOG=experimentos/logs/etapa4.log
ESPERA_S=${1:-300}
PY=.venv/bin/python
TS=$(date +%Y%m%d-%H%M)
BKP=experimentos/resultados/backups/resultados.csv.antes-da-velocidade-$TS
cp experimentos/resultados/resultados.csv $BKP
echo "== $(date) [velocidade] agendada; backup $BKP; espera de ${ESPERA_S}s para fechar os apps" >> $LOG
sleep $ESPERA_S

# ---- pre-voo: maquina ociosa (espera ate 2 h; se nao ficar ociosa, segue e marca ATENCAO) ----
ok=0; ini=$(date +%s)
while (( $(date +%s) - ini < 7200 )); do
  pesado=$(pgrep -f "rodar.py|mlx_lm|avaliar.py|kquant.py|llama-|gptq_camada|awq_liberar|dwq_sublote|medir_pico|testar_hessiana" | wc -l | tr -d ' ')
  livre=$($PY -c "import psutil;print(int(psutil.virtual_memory().available/1e9))" 2>/dev/null)
  carga=$(sysctl -n vm.loadavg | awk '{print $2}')
  if (( pesado == 0 )) && (( livre >= 10 )) && (( $(echo "$carga < 2.0" | bc) )); then ok=$((ok+1)); else ok=0; fi
  (( ok >= 2 )) && break
  sleep 60
done
apps=$(ps -axo comm | grep "/Applications/" | sed -E 's#.*/Applications/([^/]+)\.app/.*#\1#' | sort -u | tr '\n' ',' | sed 's/,$//')
mem=$($PY -c "import psutil;v=psutil.virtual_memory();s=psutil.swap_memory();print(f'{v.available/1e9:.1f} GB livres de {v.total/1e9:.1f}; swap {s.used/1e9:.1f} GB')" 2>/dev/null)
bat=$(pmset -g batt | sed -n 2p | sed 's/^ *-InternalBattery-0 ([^)]*)[[:space:]]*//')
if (( ok >= 2 )); then cond="ociosa"; else cond="ATENCAO: maquina NAO ficou ociosa em 2 h (medidas podem estar contaminadas)"; fi
echo "== $(date) [velocidade] pre-voo: $cond | carga $(sysctl -n vm.loadavg) | $mem | bateria: $bat | apps abertos: ${apps:-nenhum}" >> $LOG

# ---- controle de contaminacao (o app Claude fica aberto por decisao do usuario) ----
# Mede a MESMA configuracao (Qwen3-0,6B RTN 4 bits) 3 vezes antes e 3 depois da fase (sem gravar no CSV) e amostra, a cada
# 30 s, o CPU do app Claude e o total do sistema. A deriva entre os controles e o CPU do app permitem avaliar o efeito dele.
CTRL=experimentos/modelos/Qwen3-0.6B-rtn-4b-g64-s0
controle() {  # controle <rotulo>
  for i in 1 2 3; do
    $PY experimentos/avaliar.py $CTRL --fase velocidade 2>/dev/null | tail -1 | \
      $PY -c "import sys,json;d=json.loads(sys.stdin.read());print('[velocidade:controle] $1 #$i decode %.1f tok/s prefill %.0f tok/s'%(d['decode_tps'],d['prefill_tps']))" >> $LOG 2>&1
  done
}
echo "== $(date) [velocidade] controle ANTES" >> $LOG
controle antes
AMOSTRAS=experimentos/logs/velocidade-amostras.log
: > $AMOSTRAS
( while true; do
    claude=$(ps -axo %cpu,comm | grep -i "Claude" | grep -v grep | awk '{s+=$1} END{printf "%.1f", s+0}')
    total=$(ps -A -o %cpu | awk 'NR>1{s+=$1} END{printf "%.1f", s+0}')
    echo "$(date +%H:%M:%S) claude_cpu=$claude total_cpu=$total" >> $AMOSTRAS
    sleep 30
  done ) &
SAMP=$!

# ---- MLX ----
echo "== $(date) [velocidade] MLX: matriz.py --fase velocidade" >> $LOG
$PY experimentos/matriz.py --fase velocidade >> $LOG 2>&1
# ---- K-quants ----
echo "== $(date) [velocidade] K-quants: kquant_velocidade.py" >> $LOG
$PY experimentos/kquant_velocidade.py 2>&1 | sed 's/^/[velocidade:kquant] /' >> $LOG

kill $SAMP 2>/dev/null
echo "== $(date) [velocidade] controle DEPOIS" >> $LOG
controle depois
$PY experimentos/resumo_contaminacao.py >> $LOG 2>&1

# ---- integridade: as colunas de qualidade e os status de falha nao podem ter mudado ----
BKP=$BKP $PY - >> $LOG 2>&1 <<'E'
import csv, os
CHAVE = ["modelo", "metodo", "bits", "grupo", "seed"]
k = lambda r: tuple(r.get(c, "") for c in CHAVE)
antes = {k(r): r for r in csv.DictReader(open(os.environ["BKP"]))}
depois = {k(r): r for r in csv.DictReader(open("experimentos/resultados/resultados.csv"))}
fixas = ["ppl_wikitext2", "acc_media", "tamanho_gb", "tempo_quant_s", "data", "commit", "impl", "mlx_lm", "mlx", "python"]
difs = []
for ch, a in antes.items():
    d = depois.get(ch)
    if d is None:
        difs.append(f"{ch}: linha sumiu"); continue
    for c in fixas + [c for c in a if c.startswith("acc_")]:
        if a.get(c, "") != d.get(c, ""):
            difs.append(f"{ch[0].split('/')[-1]} {ch[1]} {ch[2]}b: {c} '{a.get(c,'')}' -> '{d.get(c,'')}'")
    sa, sd = a.get("status", ""), d.get("status", "")
    if sa.startswith("falhou") and sd != sa:
        difs.append(f"{ch[0].split('/')[-1]} {ch[1]} {ch[2]}b: status '{sa}' -> '{sd}'")
    if sa == "qualidade" and sd not in ("qualidade", "completo"):
        difs.append(f"{ch[0].split('/')[-1]} {ch[1]} {ch[2]}b: status '{sa}' -> '{sd}'")
com = sum(1 for r in depois.values() if r.get("decode_tps"))
err = sum(1 for r in depois.values() if r.get("erro_velocidade"))
inval = sum(1 for r in depois.values() if r.get("status", "").startswith("falhou:invalido"))
print(f"[INTEGRIDADE] {'OK' if not difs else 'FALHOU'}: {len(difs)} diferencas nas colunas de qualidade/status | "
      f"linhas com velocidade MLX: {com} | erros de velocidade: {err} | ainda invalidas: {inval}")
for x in difs[:30]:
    print("[INTEGRIDADE]   ", x)
kq = "experimentos/resultados/kquant-velocidade.csv"
print(f"[INTEGRIDADE] K-quants com velocidade: {sum(1 for _ in csv.DictReader(open(kq))) if os.path.exists(kq) else 0}")
E
echo "== $(date) [velocidade] fim da fase de velocidade" >> $LOG
