#!/bin/zsh
# Rodada 2 da fase de velocidade (H4) com o app Claude FECHADO (a rodada 1, de 26/10, teve o app aberto: CPU media 62%).
#  1) arquiva a rodada 1 (CSV completo, K-quants, amostras de CPU e tabela so com as colunas de velocidade);
#  2) espera ESPERA_S (padrao 300 s) para o usuario fechar o app; 3) zera as colunas de velocidade em resultados.csv e move
#  kquant-velocidade.csv (a rodada 1 fica preservada); 4) chama rodar_velocidade.sh (pre-voo, controles, 59 jobs MLX,
#  K-quants, integridade). A rodada 2 passa a ser a de referencia; a 1 serve de comparacao (efeito do app).
#   nohup caffeinate -i -s zsh experimentos/repetir_velocidade_app_fechado.sh [espera_s=300] &
set -u
cd "$(dirname "$0")/.."
LOG=experimentos/logs/etapa4.log
ESPERA_S=${1:-300}
PY=.venv/bin/python
BK=experimentos/resultados/backups
TS=$(date +%Y%m%d-%H%M)
cp experimentos/resultados/resultados.csv $BK/resultados.csv.rodada1-velocidade-app-aberto-$TS
cp experimentos/resultados/kquant-velocidade.csv $BK/kquant-velocidade.csv.rodada1-app-aberto-$TS
cp experimentos/logs/velocidade-amostras.log $BK/velocidade-amostras.rodada1-app-aberto.log
$PY experimentos/preparar_rodada2_velocidade.py arquivar >> $LOG 2>&1
echo "== $(date) [velocidade-r2] rodada 1 arquivada; rodada 2 (app fechado) comeca apos ${ESPERA_S}s" >> $LOG
sleep $ESPERA_S
$PY experimentos/preparar_rodada2_velocidade.py zerar >> $LOG 2>&1
zsh experimentos/rodar_velocidade.sh 0
echo "== $(date) [velocidade-r2] fim da rodada 2 (app fechado)" >> $LOG
