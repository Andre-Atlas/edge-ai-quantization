# H3 em escala log (analise secundaria, pos-hoc)

D = ln(PPL_q/PPL_fp) em nats/token; S = (D2-D3)-(D3-D4); superlinear se IC95% de S > 0.

| Modelo | Metodo | D4 | D3 | D2 | 4->3 | 3->2 | S | IC95% de S | Superlinear |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | rtn | 0.149 | 1.088 | 10.483 | 0.94 | 9.395 | 8.455 | [8.361, 8.55] | sim |
| Qwen3-0.6B | gptq | | | | | | | sem [3] bits: nao se calcula o incremento | |
| Qwen3-0.6B | awq | 0.123 | 0.466 | 10.123 | 0.342 | 9.657 | 9.314 | [8.877, 9.74] | sim |
| Qwen3-0.6B | dwq | 0.109 | 0.611 | 6.136 | 0.502 | 5.525 | 5.023 | [4.956, 5.096] | sim |
| Qwen3-1.7B | rtn | 0.156 | 0.897 | 10.568 | 0.741 | 9.671 | 8.93 | [8.812, 9.047] | sim |
| Qwen3-1.7B | gptq | | | | | | | sem [3] bits: nao se calcula o incremento | |
| Qwen3-1.7B | awq | 0.03 | 0.235 | 7.967 | 0.205 | 7.732 | 7.527 | [7.049, 8.008] | sim |
| Qwen3-1.7B | dwq | 0.069 | 0.44 | 5.587 | 0.372 | 5.147 | 4.776 | [4.723, 4.828] | sim |
| SmolLM3-3B | rtn | 0.16 | 2.396 | 22.903 | 2.237 | 20.507 | 18.271 | [17.093, 19.459] | sim |
| Llama-3.2-3B | rtn | 0.071 | 0.447 | 5.963 | 0.376 | 5.516 | 5.14 | [5.063, 5.215] | sim |
| Qwen3-4B | rtn | 0.06 | 0.427 | 6.766 | 0.367 | 6.339 | 5.972 | [5.897, 6.047] | sim |

**RTN, media de S sobre 5 modelo(s):** 9.356 (IC95% [9.117, 9.596]).

## Criterio original (L = 1 - Q), RTN, sem alteracao

| Modelo | L4 | L3 | L2 | 4->3 | 3->2 | Teto do 3->2 |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | 0.138 | 0.663 | 1.0 | 0.525 | 0.337 | 0.337 |
| Qwen3-1.7B | 0.145 | 0.592 | 1.0 | 0.448 | 0.408 | 0.408 |
| SmolLM3-3B | 0.148 | 0.909 | 1.0 | 0.761 | 0.091 | 0.091 |
| Llama-3.2-3B | 0.068 | 0.361 | 0.997 | 0.292 | 0.637 | 0.639 |
| Qwen3-4B | 0.058 | 0.348 | 0.999 | 0.289 | 0.651 | 0.652 |

## Segunda metade: reducao de L em 3 bits (melhor entre GPTQ e DWQ disponiveis) contra o RTN

| Modelo | Melhor | L_melhor/L_RTN | IC95% | Reduz >= 20% |
|---|---|---|---|---|
| Qwen3-0.6B | dwq (de ['dwq']) | 0.689 | [0.683, 0.695] | sim |
| Qwen3-1.7B | dwq (de ['dwq']) | 0.601 | [0.588, 0.615] | sim |
