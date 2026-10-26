#!/bin/zsh
# GPTQ do Llama-3.2-3B e do Qwen3-4B nas larguras que faltam (8 e 2 bits; 3 bits esta excluido pelo protocolo v2), em serie e sozinho
# na maquina, camada a camada (gptq_camada.py, escolhido por matriz.py quando o pico padrao nao cabe). O GPTQ de 4 bits desses
# modelos deu NaN (Hessiana em bf16); 8 e 2 bits devem dar o mesmo: o objetivo e registrar o resultado, valido ou invalido.
# Ao fim confere a perplexidade de cada um e marca como invalido (falhou:invalido-nan) os que derem NaN. Job de uso unico no launchd.
set -u
cd "$(dirname "$0")/.."
LOG=experimentos/logs/etapa4.log
echo "== $(date) [gptq-restantes] iniciando GPTQ 8 e 2 bits (Llama-3.2-3B e Qwen3-4B), em serie" >> $LOG
.venv/bin/python experimentos/matriz.py --fase qualidade --metodos gptq --somente meta-llama/Llama-3.2-3B Qwen/Qwen3-4B --workers 1 >> $LOG 2>&1
.venv/bin/python experimentos/conferir_nan.py >> $LOG 2>&1
echo "== $(date) [gptq-restantes] fim" >> $LOG
rm -f ~/Library/LaunchAgents/com.edgeai.gptq-restantes.plist
launchctl bootout gui/$(id -u)/com.edgeai.gptq-restantes 2>/dev/null
