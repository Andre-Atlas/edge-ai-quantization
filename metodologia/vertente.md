# Vertente do artigo

**Estudo experimental comparativo de quantização pós-treino (PTQ) de pequenos LLMs (0,6–4B) em um dispositivo de borda real: MacBook Pro Apple M5 Pro, 24 GB de memória unificada.** Eixos: qualidade × memória × velocidade, de 8 a 2 bits.

## Lacuna
A rodada 2 da literatura (#14–#27) encontrou avaliações de **desempenho e de sistema** de LLMs em Apple Silicon (ex.: Benazir et al., 2026, ACM SIGMETRICS PER; preprints de 2025–2026), então não se pode dizer que "não há avaliações". A lacuna reformulada é: **avaliações comparativas de algoritmos de PTQ (RTN, GPTQ, AWQ, DWQ, K-quants) lado a lado, com qualidade e eficiência medidas juntas, no mesmo dispositivo de memória unificada e em LLMs pequenos (≤ 4B), ainda são escassas.** A afirmação se baseia em resumos e deve ser confirmada na leitura completa (ver `literatura/analise-dos-artigos.md`).

## Hipóteses falseáveis (Popper)
Limiares **provisórios**; devem ser congelados em commit datado **antes** da rodada final.

| # | Hipótese | Refutada se |
|---|---|---|
| H1 | Em ≤4B, 4 bits com algoritmo ciente de dados (AWQ/GPTQ/DWQ) retém ≥ 95% da qualidade do modelo bf16, e RTN não retém. | RTN atinge o limiar, ou nenhum método ciente de dados o atinge. |
| H2 | A degradação por quantização é maior em modelos menores (0,6B > 1,7B > 4B) no mesmo bit-width. | A ordem não se confirma. |
| H3 | Abaixo de 4 bits (3 e 2) a queda é superlinear, e GPTQ/DWQ a reduzem em relação ao RTN. | A queda é linear, ou os métodos não diferem do RTN. |
| H4 | A redução de memória converte-se em ganho de *tokens*/s na decodificação aproximadamente proporcional ao bit-width. | O ganho medido fica muito abaixo do teórico (limiar a fixar). |

"Qualidade" precisa de uma definição operacional única antes da rodada final (ex.: razão perplexidade FP/quant e média normalizada de benchmarks de `lm-eval`). Isso está pendente.

## Algoritmos
RTN (baseline), GPTQ, AWQ, DWQ (mlx-lm) e K-quants/imatrix (llama.cpp). SmoothQuant, QLoRA, BitNet e MCUNetV2 entram só no referencial (justificativa em [`../literatura/analise-dos-artigos.md`](../literatura/analise-dos-artigos.md)).

## Modelos
Núcleo Qwen3-0.6B/1,7B/4B (Apache-2.0) + SmolLM3-3B (Apache-2.0) + Llama-3.2-3B (licença Llama 3.2, acesso por aprovação). Opcionais: Phi-4-mini, Gemma 4, BitNet b1.58 2B.

## Limitações já assumidas
- Um único dispositivo (Apple Silicon): resultados de velocidade e memória não se generalizam a CUDA, NPU ou MCU.
- Python 3.14.9 (3.15 não tem wheels do PyTorch; SentencePiece não tem wheel para 3.15 e a compilação falhou no ligador (`ld: library System not found`; causa provável: licença do Xcode não aceita, não confirmada)).
- Calibração: texto público `calibration_v5` (usado pelo mlx-lm); avaliação em WikiText-2 (teste): conjuntos distintos.
- O sistema avisa que a licença do Xcode não foi aceita (`sudo xcodebuild -license`); isso pode impedir compilar `llama.cpp`. Plano: usar o binário oficial pré-compilado.
