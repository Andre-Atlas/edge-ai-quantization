# Experimentos

Pipeline para quantizar e avaliar LLMs pequenos no Apple Silicon com [mlx-lm](https://github.com/ml-explore/mlx-lm).

## Ambiente
Python **3.14.9** gerenciado por [`uv`](https://docs.astral.sh/uv/) (0.13.5). Versões travadas em `uv.lock`.

```bash
uv sync
```

Python 3.15 não é usado: o `torch` 2.14.1 ainda não publica wheels para ele.

## Rodar
```bash
uv run python experimentos/rodar.py --modelo Qwen/Qwen3-0.6B --metodo fp
uv run python experimentos/rodar.py --modelo Qwen/Qwen3-0.6B --metodo rtn  --bits 4
uv run python experimentos/rodar.py --modelo Qwen/Qwen3-0.6B --metodo gptq --bits 4
uv run python experimentos/rodar.py --modelo Qwen/Qwen3-0.6B --metodo awq  --bits 4
uv run python experimentos/rodar.py --modelo Qwen/Qwen3-0.6B --metodo dwq  --bits 4
```
Cada execução acrescenta uma linha em `resultados/resultados.csv` e apaga o modelo quantizado (use `--manter` para guardá-lo em `modelos/`, ignorado pelo git).

## Métricas
- `ppl_wikitext2`: perplexidade no conjunto de **teste** do WikiText-2 (raw), janelas de 2048 tokens sem sobreposição ([`avaliar.py`](avaliar.py)).
- `prefill_tps` / `decode_tps` (± desvio-padrão): prompt de 512 tokens, 128 gerados, 5 repetições após 1 de aquecimento.
- `pico_memoria_gb`: pico de memória de GPU/unificada medido pelo MLX durante a geração.
- `tamanho_gb`: tamanho em disco do modelo.

## Cuidados
- Calibração (AWQ/GPTQ): `calibration_v5` (baixado automaticamente pelo mlx-lm), distinto do conjunto de avaliação.
- **Piloto** (Qwen3-0.6B, 4 bits, uma execução cada): ainda não há repetições nem benchmarks de tarefas; os números servem para validar o pipeline, não para conclusões.
- Energia (`powermetrics`, exige `sudo`) ainda não está instrumentada.
