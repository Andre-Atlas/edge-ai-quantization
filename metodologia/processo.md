# Processo de desenvolvimento em etapas

Cada etapa só começa quando a anterior passa na sua validação. O estado de cada uma é atualizado aqui e no painel (`site/`).

| Etapa | Entrega | Validação | Estado |
|---|---|---|---|
| 0. Ambiente e piloto | `uv` + Python 3.14.9, mlx-lm 0.32.2; pipeline MLX; piloto no Qwen3-0.6B (4 bits) | FP e os 4 métodos rodam de ponta a ponta | ✅ concluída |
| 1. Protocolo congelado | [`protocolo.md`](protocolo.md), tag `protocolo-v1` | limiares de H1–H4 fixados em commit datado antes da rodada final | ✅ tag `protocolo-v1` |
| 2. Pipeline robusto | `rodar.py`, `avaliar.py`, `matriz.py`, `testes.py` | `experimentos/testes.py` passa (determinismo, monotonicidade, 8 bits ≈ FP, tamanho, lm-eval) | ✅ 8/8 testes ([log](../experimentos/resultados/testes-protocolo-v1.log)) |
| 3. llama.cpp (K-quants) | `experimentos/kquant.py` | Q8_0 ≤ 1% do F16; PPL monotônica entre os tipos | ✅ validado no Qwen3-0.6B (Q8_0 +0,2%) |
| 4. Matriz completa | `resultados/resultados.csv` (89 jobs de qualidade + velocidade isolada) | checagens por job; falhas marcadas, sem interromper a fila | ⏳ |
| 5. Análise das hipóteses | `analise.py` → `hipoteses.json` | veredito só pelos critérios do protocolo; IC que cruza o limiar = inconclusiva | ⏳ |
| 6. Painel (GitHub Pages) | `site/` publicado | abre em desktop e celular, sem erro no console, números iguais ao CSV | 🟡 validado localmente; publicação no Pages |
| 7. Literatura e redação | rodada 2 de busca, fichas, rascunho pt-BR | cada afirmação aponta para uma linha do CSV; cada citação com DOI/arXiv conferido | 🟡 rodada 2 feita (14 itens, metadados conferidos); fichas e redação pendentes |

## Requisitos e resultados esperados por etapa

**Etapa 1.** Requisito: definir "qualidade" (`Q = PPL_fp/PPL_q`, mais acurácia zero-shot em 5 tarefas), os critérios de refutação de H1–H4 e a grade experimental. Resultado: documento + tag. O piloto já visto é declarado no próprio protocolo.

**Etapa 2.** Requisitos: fases separadas (*qualidade*, que pode rodar em paralelo, e *velocidade*, sempre isolada), retomada, gravação segura com *lock*, semente e hash do git em cada linha, NLL por janela salva para *bootstrap*. Resultado: pipeline que reproduz o piloto com as novas colunas.

**Etapa 3.** Requisito: llama.cpp compilado com Metal e o mesmo texto de calibração. Resultado: tipos Q8_0, Q4_K_M, Q3_K_M, Q2_K com PPL e KL contra o F16. Só razões contra o próprio F16 são comparadas ao MLX.

**Etapa 4.** Requisitos: 5 modelos × (FP + RTN/GPTQ/AWQ/DWQ × 8/4/3/2 bits) + réplicas de semente do AWQ; Llama-3.2-3B exige `hf auth login` do autor. Execução: qualidade em paralelo com guarda de memória; velocidade e memória depois, uma por vez, com a máquina ociosa (processos concorrentes dividem GPU e banda de memória e invalidariam a H4).

**Etapa 5.** Requisito: bootstrap de 10 000 reamostragens sobre as janelas de PPL; fronteira de Pareto. Resultado: `hipoteses.json` com corroborada / refutada / inconclusiva e a evidência de cada uma.

**Etapa 6.** Requisito: site estático gerado a partir do CSV. Resultado: contexto, lógica dos testes, gráficos interativos, hipóteses e pontos de melhoria.

**Etapa 7.** Requisito: rodada 2025–2026 (DWQ, K-quants, Apple Silicon, KV-cache, MXFP4). Resultado: texto em pt-BR (ABNT) com tabelas e figuras geradas pelos scripts, reutilizáveis na versão em inglês.

## Resultados obtidos até aqui
**Validação do pipeline (Etapa 2, Qwen3-0.6B):** RTN repetido dá PPL idêntica (24,3302); PPL monotônica 21,04 (8b) ≤ 24,33 (4b) ≤ 62,26 (3b) ≤ 748 573 (2b, colapso do RTN); 8 bits a 0,35% do FP; tamanhos a ≤ 6% do teórico; lm-eval funcional.

**K-quants (Etapa 3, Qwen3-0.6B, razão PPL_F16/PPL):** Q8_0 0,998 · Q4_K_M 0,939 (5,15 bits/peso) · Q3_K_M 0,861 · Q2_K 0,497.

**Literatura (Etapa 7):** a rodada 2 encontrou avaliações de sistema em Apple Silicon; a lacuna foi reformulada para *comparação de algoritmos de PTQ com qualidade e eficiência juntas* (ver `vertente.md`).

Piloto (Qwen3-0.6B, 4 bits, 1 execução, `resultados/piloto-v0.csv`): PPL bf16 20,97 · GPTQ 22,87 · DWQ 23,44 · AWQ 23,99 · RTN 24,33; decodificação ≈ 2× mais rápida e modelo ≈ 3,4× menor. Serve para validar o pipeline, não para concluir.
