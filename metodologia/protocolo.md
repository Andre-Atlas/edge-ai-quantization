# Protocolo experimental (pré-registro) — versão 1

> Congelado na tag git `protocolo-v1`. Qualquer mudança depois da tag exige uma nova versão (`protocolo-v2`), registrada com o motivo, e os resultados passam a indicar a versão usada. Isso é a aplicação prática do critério de Popper: o que conta como refutação é decidido **antes** de olhar os resultados finais.

**Transparência:** o piloto (Qwen3-0.6B, 4 bits, uma execução; `experimentos/resultados/piloto-v0.csv`) foi visto **antes** de escrever estes limiares. Eles foram escolhidos a partir da literatura (95% de retenção é um patamar comum em relatos de quantização) e não ajustados para o resultado do piloto; ainda assim, o leitor deve saber que o piloto existiu.

## 1. Definições operacionais

**Qualidade primária.** `Q = PPL_fp / PPL_q`, no conjunto de teste do WikiText-2 (raw), janelas de 2048 tokens sem sobreposição. `Q = 1` equivale ao modelo bf16; valores menores indicam degradação. O `PPL_fp` é do mesmo modelo em bf16, medido pelo mesmo código.

**Qualidade secundária.** Acurácia zero-shot média em ARC-Easy, ARC-Challenge, HellaSwag, PIQA e WinoGrande (lm-evaluation-harness via `mlx_lm evaluate`, sem *chat template*; `acc_norm` quando existe, senão `acc`). `Q_acc = acc_media_q / acc_media_fp`.

**Eficiência de velocidade.** `E = (decode_q / decode_fp) / (tamanho_fp / tamanho_q)`: fração do ganho teórico (decodificação limitada por banda de memória) que se realiza na prática.

**Intervalos.** IC de 95% por *bootstrap* sobre as janelas de 2048 tokens (10 000 reamostragens) para `Q`; erro-padrão do lm-eval para as acurácias; desvio-padrão de 5 repetições para velocidade. O veredito só é "corroborada" ou "refutada" quando o IC inteiro cai de um lado do limiar; caso contrário, **"inconclusiva"**.

## 2. Hipóteses e critérios de refutação

Métodos "cientes de dados": GPTQ, AWQ, DWQ. Modelos: Qwen3-0.6B, Qwen3-1.7B, SmolLM3-3B, Llama-3.2-3B, Qwen3-4B (5 modelos).

| # | Hipótese | Corroborada se | Refutada se |
|---|---|---|---|
| H1 | Em 4 bits, algum método ciente de dados retém ≥ 95% da qualidade (`Q ≥ 0,95`) e o RTN não. | Em ≥ 4 dos 5 modelos, o melhor método ciente de dados tem `Q ≥ 0,95` **e** o RTN tem `Q < 0,95`. | Em ≥ 3 dos 5 modelos, o RTN tem `Q ≥ 0,95`, **ou** em ≥ 3 modelos nenhum método ciente de dados atinge `Q ≥ 0,95`. |
| H2 | A degradação é maior em modelos menores, mesma família (Qwen3). | Para ≥ 75% das células (método × {4, 3 bits}), `Q(0,6B) < Q(1,7B) < Q(4B)` com ICs sem sobreposição entre vizinhos. | Para < 50% das células a ordem se confirma. |
| H3 | Abaixo de 4 bits a queda é superlinear e métodos que aprendem (GPTQ, DWQ) a reduzem em relação ao RTN. | Com `L = 1 − Q`: `L(2) − L(3) > L(3) − L(4)` na média dos modelos (para RTN) **e** em 3 bits `L_GPTQ` ou `L_DWQ` é ≥ 20% menor que `L_RTN` em ≥ 4 dos 5 modelos. | Os incrementos de L não são crescentes, **ou** nem GPTQ nem DWQ reduz `L` em ≥ 20% no 3 bits em ≥ 3 modelos. |
| H4 | O ganho de memória vira ganho de velocidade quase proporcional. | `E ≥ 0,6` em 4 bits para os modelos de 3–4B (SmolLM3, Llama-3.2-3B, Qwen3-4B). | `E < 0,4` para esses modelos. |

Resultados fora dessas faixas (ou com IC que cruza um limiar) são reportados como inconclusivos.

## 3. Grade experimental

- **Bits:** 8, 4, 3, 2. **Group size:** 64 (todos os métodos baseados em MLX). Escala e *bias* por grupo em fp16.
- **Métodos:** RTN, GPTQ, AWQ, DWQ (mlx-lm 0.32.2) e K-quants/imatrix (llama.cpp; razões contra o F16 do próprio llama.cpp).
- **Sementes:** 1 (semente 0) na matriz completa. Réplicas com sementes 1 e 2 para o AWQ (o único que sorteia amostras de calibração) a 4 bits em Qwen3-0.6B e 1,7B, para estimar a variância entre sementes. RTN é determinístico.
- **Calibração:** `calibration_v5` (texto público usado pelo mlx-lm), com os padrões de número de amostras e comprimento de cada método do mlx-lm 0.32.2 (AWQ: 128 amostras × 512 tokens sorteadas do texto; GPTQ: todas as amostras disponíveis × 512 tokens, logo a semente quase não o afeta; DWQ: padrão da ferramenta); distinto do WikiText-2.
- **Velocidade:** prompt de 512 tokens, 128 gerados, 5 repetições após 1 de aquecimento; **máquina sem outros jobs**. Pico de memória medido pelo MLX.
- **Ambiente:** Python 3.14.9, `uv.lock`; cada linha do CSV grava o commit git.

## 4. Ameaças à validade já conhecidas
- Um único dispositivo (Apple Silicon M5 Pro, 24 GB): velocidade e memória não se generalizam a CUDA, NPU ou MCU.
- Perplexidade e acurácia zero-shot não esgotam a qualidade (raciocínio longo, código e geração aberta ficam de fora).
- Uma semente na matriz principal: diferenças pequenas entre métodos (< ~1 ponto de PPL) podem estar dentro do ruído de calibração; as réplicas estimam isso.
- O DWQ é aprendido com poucos dados e hiperparâmetros padrão; um ajuste fino poderia mudar seu desempenho.
- A PPL do llama.cpp não é comparável em valor absoluto à do MLX; só razões contra o próprio FP.
