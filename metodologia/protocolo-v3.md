# Protocolo experimental — versão 3 (extensão pré-registrada)

> **Status: pré-registro (tag `protocolo-v3`), feito ANTES de qualquer execução da extensão.** Não substitui a v1 (`protocolo.md`, tag `protocolo-v1`) nem a v2 (`protocolo-v2.md`, tag `protocolo-v2`): os vereditos da v2 (H1, H2 e H3 inconclusivas; H4 corroborada) **permanecem** e são reportados como resultado do estudo original. A v3 é um **segundo teste**, com dados novos e regras escritas para que o resultado seja decidível. Ela aumenta a chance de um veredito conclusivo, **sem garanti-lo**. Qualquer mudança de regra depois da tag exige uma v4. O estado de execução é registrado em `metodologia/caderno-de-laboratorio.md`.

## 1. Por que uma v3

Com os dados da v2, **nenhum dado ainda possível muda H1, H2 ou H3**: as três são inconclusivas por desenho.

- **H1.** Há 2 modelos com sucesso (Qwen3-1,7B e Llama-3.2-3B) e 2 com "nenhum método ciente ≥ 0,95" (Qwen3-0,6B e Qwen3-4B). A corroboração exige 4 de 4 e a refutação exige 3.
- **H2.** Só há 2 células (RTN × {4, 3 bits} nos três Qwen3). Uma fração de 1 em 2 (50%) é inconclusiva por construção.
- **H3.** Na 2ª metade, só 2 modelos são avaliáveis (DWQ a 3 bits). Há três razões:
  - o GPTQ do mlx-lm não aceita 3 bits;
  - o GPTQ é inválido (NaN) nos modelos de 3–4B;
  - o DWQ não cabe na memória nesses modelos.

Mudar as regras olhando esses dados seria ajuste pós-hoc. A v3 faz três coisas:

- acrescenta **dados novos**: 4 modelos e um método corrigido;
- converte os limiares da v1 em **proporções**;
- decide cada modelo **pelo IC95%**.

Tudo isso fica fixado antes de executar.

## 2. O que muda

### 2.1 Novos modelos (fora da amostra)
| Modelo | Papel | Métodos |
|---|---|---|
| Qwen/Qwen2.5-0.5B | 2ª família (H2), H1, H3 | FP, RTN 4/3/2, AWQ 4/3, GPTQ-fp32 4/3, DWQ 4/3; réplicas AWQ 4 bits (sementes 1 e 2) |
| Qwen/Qwen2.5-1.5B | 2ª família (H2), H1, H3 | idem |
| Qwen/Qwen2.5-3B | 2ª família (H2), H1, H3 | FP, RTN 4/3/2, AWQ 4/3, GPTQ-fp32 4/3 (DWQ excluído por memória, como nos demais modelos de 3–4B) |
| meta-llama/Llama-3.2-1B | H1, H3 | FP, RTN 4/3/2, AWQ 4/3, GPTQ-fp32 4/3, DWQ 4/3 |

Critério de escolha, fixado antes de ver qualquer resultado desses modelos:
- base (não instruído);
- ≤ 3B parâmetros;
- arquitetura com AWQ suportado no mlx-lm 0.32.2 (`qwen2`, `llama`);
- sem janela deslizante, o que exclui o Gemma-3-1B, porque o `gptq_camada.py` não a suporta;
- uma família com três tamanhos, para replicar a H2.

### 2.2 GPTQ corrigido (`gptq_fp32`) — método próprio da v3
- É o `gptq_camada.py` com `--hessiana fp32`: o produto xᵀx é calculado em float32. É a causa do NaN demonstrada na v2 (seção 4).
- Aceita **3 bits**. O empacotamento do mlx-lm só aceita 2, 4 e 8 bits, então o `empacotar()` usa o mesmo formato do `mx.quantize`, conferido bit a bit em matrizes aleatórias para 2, 3, 4, 6 e 8 bits.
- Tem nome próprio no CSV (`gptq_fp32`, coluna `impl` = "gptq_camada.py (Hessiana fp32, v3)") e **não** é o GPTQ padrão do mlx-lm. A falha do padrão continua reportada como falha de método.
- Roda nos **9 modelos**: os 5 da v2 e os 4 novos.

### 2.3 Grade mínima suficiente
Só o que entra em alguma hipótese:
- FP;
- RTN a 4, 3 e 2 bits;
- AWQ, GPTQ-fp32 e DWQ (onde couber) a 4 e 3 bits.

O 8 bits fica de fora porque não entra em nenhuma hipótese. A H4 já foi decidida na v2, e a velocidade dos modelos novos é opcional (descritiva).

A grade está em `experimentos/matriz-v3.json` (`matriz.py --config matriz-v3.json`). Ferramenta e hardware são os da v2: mlx-lm 0.32.2, Apple M5 Pro com 24 GiB. Avaliação idêntica: WikiText-2 com janelas de 2048 tokens e as 5 tarefas zero-shot.

### 2.4 Variância da calibração
- O AWQ tem réplicas de semente a 4 bits (sementes 0, 1 e 2) no Qwen3-0,6B e 1,7B (v2) e no Qwen2.5-0.5B e 1.5B (v3). Com elas se estima `σ_cal`, o desvio-padrão de Q entre sementes, combinado entre os modelos.
- Para métodos com calibração (AWQ, GPTQ-fp32, DWQ), a meia-largura do IC de Q passa a ser `√(h_boot² + (1,96·σ_cal)²)`.
- Em diferenças entre dois métodos calibrados, soma-se `2·σ_cal²` à variância.

## 3. Regras de decisão (v3)

Notação: `Q = PPL_fp / PPL_q`, com IC95% por bootstrap pareado sobre as janelas (10 000 reamostragens, semente 0). Para os métodos calibrados, o IC é ampliado como na seção 2.4.

### H1 — "em 4 bits, algum método ciente retém ≥ 95% e o RTN não"
- **Métodos cientes por modelo:** os válidos a 4 bits entre GPTQ padrão, GPTQ-fp32, AWQ e DWQ. O **melhor** é o de maior Q.
- **Avaliável:** o modelo tem RTN 4 bits válido e pelo menos um método ciente válido.
- **Sucesso:** IC_inf(melhor) ≥ 0,95 **e** IC_sup(RTN) < 0,95.
- **Contrário:** IC_inf(RTN) ≥ 0,95, **ou** IC_sup < 0,95 em todos os métodos cientes.
- **Indeterminado:** qualquer outro caso.
- **Veredito:**
  - **corroborada** se sucessos ≥ 80% dos avaliáveis (a v1 pedia 4 de 5);
  - **refutada** se contrários ≥ 60% (a v1 pedia 3 de 5);
  - **inconclusiva** nos demais casos;
  - exige N ≥ 5 avaliáveis, senão "não avaliável".

### H2 — "a degradação é maior em modelos menores da mesma família"
- **Célula:** família (Qwen3: 0,6B/1,7B/4B; Qwen2.5: 0.5B/1.5B/3B) × método (RTN, AWQ, GPTQ-fp32) × bits (4, 3). São até 12 células.
- **Célula válida:** os três tamanhos têm o dado válido.
- **Ordem confirmada:** as diferenças pareadas por bootstrap ΔQ(médio − pequeno) e ΔQ(grande − médio) têm **IC_inf > 0**. As janelas são as mesmas porque a família usa o mesmo tokenizador; isso é conferido pelo número de janelas.
- **Veredito:**
  - **corroborada** se a ordem se confirma em ≥ 75% das células válidas;
  - **refutada** se em < 50% (limiares da v1);
  - **inconclusiva** nos demais casos;
  - exige ≥ 6 células válidas.

### H3 — "abaixo de 4 bits a queda é superlinear, e métodos que aprendem a reduzem"
- **1ª metade (RTN, por modelo), na escala de NLL** (ver a nota abaixo):
  - `D_b = média(NLL_q,b − NLL_fp)` e `S = (D₂ − D₃) − (D₃ − D₄)`.
  - **Superlinear** se IC_inf(S) > 0; **contrário** se IC_sup(S) < 0.
- **2ª metade (3 bits, por modelo):** `L = 1 − Q`, e o melhor entre GPTQ-fp32 e DWQ válidos.
  - **Reduz** se IC_sup(L_melhor / L_RTN) < 0,8; **contrário** se IC_inf(L_melhor / L_RTN) > 0,8.
- **Veredito:**
  - **corroborada** se superlinear ≥ 80% **e** reduz ≥ 80% dos avaliáveis de cada metade;
  - **refutada** se contrários ≥ 60% em qualquer das metades;
  - **inconclusiva** nos demais casos;
  - exige N ≥ 5 em cada metade.
- **Nota sobre a escala:** na v2, a escala de NLL era uma análise secundária pós-hoc, motivada pelo teto de `L = 1 − Q` aos 2 bits. Na v3 ela é **escolhida antes** dos dados novos e passa a ser o critério da 1ª metade. O critério em L da v1 continua calculado e reportado.

### Amostra e transparência
- **Análise primária:** todos os modelos avaliáveis. São os 9 na H1 e na H3 e as duas famílias na H2. O SmolLM3 passa a contar se o seu GPTQ-fp32 for válido.
- **Robustez:** só os 4 modelos novos (fora da amostra), reportados lado a lado com a mesma classificação por modelo. Como N = 4 < 5, isso não gera veredito.
- **Já visto antes desta versão:** todos os resultados da v2, incluindo os vereditos, o RTN, o AWQ, o DWQ e o GPTQ padrão dos 5 modelos antigos. Ainda **não** foi visto nenhum resultado dos 4 modelos novos nem do GPTQ-fp32 de qualquer modelo; o único diagnóstico do fp32 foi no Qwen3-0,6B a 4 bits, na v2, seção 7 A, com razão de Q de 1,0143.
- **Teste do código nos dados da v2, antes da tag:** o `analise_v3.py` foi rodado uma vez sobre os dados da v2 para conferir que funciona. Resultado:
  - σ_cal = 0,0065;
  - H1 com 1 sucesso, 2 contrários e 1 indeterminado entre 4 avaliáveis;
  - H2 com 1 de 4 células válidas;
  - 1ª metade da H3 superlinear em 5 de 5 (já conhecido pela análise pós-hoc da v2);
  - todos "não avaliável" por N insuficiente.

  As regras acima **já estavam escritas** e não foram alteradas depois disso.
- **Nenhum limiar novo foi escolhido:** são os da v1 convertidos em proporção (4/5 → 80%, 3/5 → 60%, 75% e 50% das células, 20% de redução, 0,95 de retenção).

## 4. Validação antes da fila (fim de semana 1)
1. **Regressão:** o `gptq_camada.py` com `--hessiana bf16` a 4 bits no Qwen3-0,6B continua com tensores idênticos ao GPTQ padrão (`comparar_gptq.py`). Isso prova que o `empacotar()` não alterou o caminho de 2, 4 e 8 bits.
2. **Sanidade do 3 bits:** GPTQ-fp32 a 3 bits no Qwen3-0,6B com PPL finita e entre a do RTN 3 bits e a do GPTQ-fp32 4 bits.
- Se 1 ou 2 falhar, a fila do GPTQ-fp32 não roda e o problema é registrado antes de qualquer outra decisão.

## 5. Cronograma e restrições
- As execuções só acontecem de **sexta 18h a segunda** (dia inteiro), pelo `rodar_fim_de_semana.sh` via `launchd`, que não inicia job novo depois de segunda 22h.
- **Fins de semana previstos:**
  - 30/10–02/11: validação, GPTQ-fp32 dos 5 modelos antigos, Qwen2.5-0.5B e Llama-3.2-1B;
  - 06–09/11: Qwen2.5-1.5B, Qwen2.5-3B e as réplicas;
  - 13–16/11: reserva.
- **Dados congelados em 16/11.** Análise e painel até 27/11; artigo até 18/12/2026.

## 6. Ameaças à validade (acréscimos)
- A análise primária mistura dados já vistos (os 5 modelos antigos) com dados novos. Por isso a robustez só com os novos é reportada ao lado.
- O GPTQ-fp32 não é o GPTQ "de prateleira" do mlx-lm. As conclusões sobre GPTQ na v3 valem para essa implementação.
- A variância da calibração é estimada só pelo AWQ e aplicada aos três métodos calibrados (suposição declarada).
- A H2 com a família Qwen2.5 mede a mesma relação em outra geração de modelos. Uma divergência entre as famílias é reportada célula a célula.

## 7. Rastreabilidade
- `experimentos/matriz-v3.json`: grade.
- `experimentos/gptq_camada.py`: `empacotar()` e `--hessiana fp32`.
- `experimentos/rodar.py`: método `gptq_fp32`.
- `experimentos/analise_v3.py`: regras desta seção, com saída em `resultados/hipoteses-v3.json`.
- `experimentos/testes_v3.py`: testes sintéticos dos ramos.
- `experimentos/rodar_fim_de_semana.sh`: orquestração.
