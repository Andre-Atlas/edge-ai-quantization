#!/bin/zsh
# Etapa 4 completa, retomavel (pode ser interrompida e relancada).
#   caffeinate -i zsh experimentos/executar_etapa4.sh <workers>
# 1) qualidade em paralelo (MLX)  2) K-quants por modelo (llama.cpp, em serie)
# 3) velocidade em serie, maquina ociosa  4) analise + painel
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
W=${1:-2}
export PATH="$HOME/.local/bin:$PATH"
echo "== $(date) qualidade (workers=$W)"
$PY experimentos/matriz.py --fase qualidade --workers "$W"
echo "== $(date) K-quants"
for m in $($PY -c "import json;print(' '.join(x['id'] for x in json.load(open('experimentos/matriz.json'))['modelos']))"); do
  if grep -q "^$m,kquant,Q2_K" experimentos/resultados/kquant.csv 2>/dev/null; then echo "kquant $m ja feito"; continue; fi
  $PY experimentos/kquant.py --modelo "$m" --tipos Q8_0,Q4_K_M,Q3_K_M,Q2_K > "experimentos/logs/kquant-${m//\//_}.log" 2>&1 \
    && echo "[ok] kquant $m" || echo "[FALHOU] kquant $m"
done
echo "== $(date) velocidade (serie, isolada)"
$PY experimentos/matriz.py --fase velocidade
echo "== $(date) analise"
$PY experimentos/analise.py
$PY experimentos/exportar_site.py
echo "== $(date) fim"
