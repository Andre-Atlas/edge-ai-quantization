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

## Memória: por que o Mac travou e como isso é evitado

**Incidente (22/10/2026, 13:34 e 13:51):** o Mac congelou duas vezes e precisou ser reiniciado durante uma medição com **3 avaliações em paralelo**. O log do kernel mostra dezenas de `IOGPUFamily ... <python3.14> failed to prepare memory status = 0xE00002BE` (sem recursos) com ~19 GB alocados na GPU, seguidas de `WindowServer ... Watchdog ... display 1 regained readiness after 12.55s`: a GPU (memória unificada) esgotou e a interface do macOS, que também depende dela, parou.

**Causas:**
1. `avaliar.py` calculava a perplexidade em lotes de 4 janelas de 2048 tokens. Com vocabulário de ~152 mil tokens, os logits em float32 + temporários ocupavam **~12 GB por processo**, mesmo num modelo de 0,6B.
2. A estimativa de memória da matriz considerava só os pesos (~3,6 GB por job) e liberou 3 jobs simultâneos (~37 GB pedidos num Mac de 24 GB).
3. O processo avaliador segurava o modelo carregado enquanto o lm-eval carregava outra cópia; o mlx-lm ainda fixa memória na GPU (`set_wired_limit`), que o macOS não consegue recuperar.
4. Com o Mac em uso normal (Teams, Outlook, Chrome, Claude, Spotlight), sobram só ~11 GB livres.

**Prevenção (já aplicada):**
- Perplexidade **uma janela por vez** com `mx.clear_cache()` a cada passo: pico medido caiu de ~12 GB para **3,7 GB** no Qwen3-0.6B, com PPL idêntica (20,9669).
- **Teto de memória por processo** (`mx.set_memory_limit` = pesos + 4 GB; ajustável por `EAQ_LIMITE_GPU_GB`) e cache do MLX limitado a 1 GB.
- Modelo liberado (`del` + `gc` + `mx.clear_cache()`) antes do lm-eval; lm-eval com `--batch-size 4`.
- DWQ com `--grad-checkpoint` (só recalcula ativações; não muda o resultado).
- `matriz.py`: **serial por padrão** (`--workers 1`); estimativa conservadora (pesos × 1,5 + 4 GB; DWQ pesos × 3 + 4 GB); orçamento de 12 GB; e **não inicia nenhum job sem memória livre real suficiente** (pesos + 6 GB), esperando e avisando no log em vez de arriscar travar.

**Recomendação de uso:** rodar a matriz com o Mac ocioso (fechar Teams, Outlook, Chrome), sobretudo para SmolLM3-3B, Llama-3.2-3B e Qwen3-4B. A fase de velocidade já exige máquina ociosa de qualquer forma. Paralelismo na mesma GPU não compensa o risco nesta máquina.
