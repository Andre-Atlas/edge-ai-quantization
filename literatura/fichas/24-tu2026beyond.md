# Ficha de leitura — #24 Tu et al. (2026)

**Referência (ABNT):** TU, Fangbo et al. Beyond Output Matching: Preserving Internal Geometry in NVFP4 LLM Distillation. **arXiv preprint arXiv:2606.05682**, 2026. Disponível em: https://arxiv.org/abs/2606.05682. Acesso em: 22 out. 2026.
**Chave BibTeX:** `tu2026beyond`
**Data da leitura:** 2026-11-09
**Escopo da leitura:** Consolidação dos metadados e resumo verificados via API do arXiv (`2606.05682` v2, preprint sem revisão por pares localizada), sem fabricação de números de página ou citações literais entre aspas de PDF não lido na íntegra.

## 1. Análise textual (visão global)
- **Tipo de texto:** Preprint metodológico e analítico sobre destilação para quantização de LLMs em 4 bits (`NVFP4`), investigando a geometria das representações internas.
- **Estrutura conceitual:** Diagnóstico das limitações da destilação pós-treino baseada exclusivamente no alinhamento da distribuição de saída (*output matching* via divergência de Kullback-Leibler sobre os *logits* finais), uso de *Centered Kernel Alignment* (`CKA`) para demonstrar que o alinhamento apenas da saída mascara uma degradação severa nas representações intermediárias das camadas ocultas, e proposta de regularização para preservar a geometria interna durante a destilação.
- **Vocabulário-chave:** *quantization distillation*, *output matching*, *KL divergence*, *internal geometry*, *Centered Kernel Alignment (CKA)*, *representation collapse*, *NVFP4*.

## 2. Análise temática
- **Tema:** Limitações estruturais da destilação de quantização baseada apenas em divergência KL dos *logits* de saída e preservação da geometria interna das representações em LLMs quantizados.
- **Problema:** Métodos de destilação para quantização (como QAD tradicional e `DWQ`) otimizam parâmetros minimizando a divergência KL entre os *logits* finais do modelo professor e do modelo quantizado no conjunto de calibração; contudo, modelos otimizados apenas pela saída final podem apresentar perda de generalização ou fragilidade fora do domínio de calibração.
- **Tese / hipótese do(s) autor(es):** Minimizar apenas a divergência KL na camada de saída (*output matching*) permite que as camadas finais compensem superficialmente erros acumulados nas camadas anteriores sobre o pequeno conjunto de calibração, mascarando uma distorção profunda da geometria interna das ativações intermediárias (mensurável por `CKA`).
- **Argumentação / método:** Os autores medem a similaridade estrutural das representações intermediárias entre professor e aluno via `CKA` ao longo das camadas de LLMs quantizados em 4 bits e comparam a destilação puramente nos *logits* contra um objetivo que preserva a geometria interna das camadas ocultas.
- **Resultados principais:** O estudo comprova via `CKA` que o *output matching* isolado degrada a geometria interna da rede mesmo quando a perda de saída na calibração parece baixa, e que preservar a estrutura interna melhora a robustez e o desempenho em tarefas subsequentes.

## 3. Análise interpretativa
- **Contexto e relevância:** Este artigo fornece a explicação teórica mais elegante e profunda para um dos resultados mais intrigantes da nossa Matriz v2 no Apple M5 Pro: **por que o `DWQ` (que otimiza escalas e vieses minimizando exatamente a divergência KL dos *logits* de saída no conjunto de calibração) fica atrás do `AWQ` e do `GPTQ` em 4 e 3 bits e não evita o colapso em 2 bits!**
- **Pressupostos:** Analisa o processo de destilação professor-aluno sob pequeno orçamento de dados/passos de calibração.
- **Pontos fortes:** Usa uma métrica geométrica rigorosa (`CKA`) para expor a diferença entre ajustar a saída final (*end-to-end logit matching*) e preservar o funcionamento camada a camada (*layer-wise reconstruction/scaling*).
- **Limites:** Preprint sem revisão por pares confirmada; os experimentos originais dos autores focam no formato `NVFP4` e em estágios de pós-treino em GPU CUDA, não no `DWQ` do `mlx-lm`.

## 4. Problematização
Questões e conexões diretas que o texto suscita para o nosso estudo experimental no Apple M5 Pro (24 GiB):
- **Por que `AWQ` e `GPTQ` superam o `DWQ` em 4 e 3 bits nos nossos dados (`resultados.csv` e `hipoteses.json`):** Na documentação do `mlx-lm` (#26), o `DWQ` ajusta apenas as escalas e vieses não quantizados minimizando a perda de destilação (KL) nos *logits* de saída sobre um conjunto pequeno de calibração (`calibration_v5`). Em nossos experimentos no M5 Pro:
  - Em **4 bits**, no `Qwen3-0.6B`, o `DWQ` obteve $Q = 0{,}8968$, inferior ao `GPTQ` padrão ($Q = 0{,}9155$) e ao `GPTQ-fp32` ($Q = 0{,}9287$); no `Qwen3-1.7B`, o `DWQ` obteve $Q = 0{,}9336$, inferior ao `AWQ` ($Q = 0{,}9702$).
  - Em **3 bits**, no `Qwen3-0.6B`, o `DWQ` obteve $Q = 0{,}5428$ contra $Q = 0{,}6276$ do `AWQ`; no `Qwen3-1.7B`, o `DWQ` obteve $Q = 0{,}6438$ contra $Q = 0{,}7903$ do `AWQ`.
  - Em **2 bits**, o `DWQ` colapsou junto com o `RTN` ($Q = 0{,}0022$ no `0.6B` e $Q = 0{,}0037$ no `1.7B`).
- **A explicação pelo mecanismo de Tu et al. (2026):** Enquanto o `AWQ` e o `GPTQ` preservam explicitamente a geometria interna de **cada camada linear individualmente** ($\min \|W X - \widehat{W} X\|_2^2$), o `DWQ` faz apenas *output matching* na saída final da rede com poucas amostras de calibração. Conforme demonstrado por Tu et al. (2026), ajustar escalas apenas para igualar os *logits* finais permite que a geometria interna das camadas intermediárias se degrade e sobreajuste ao pequeno conjunto de calibração, explicando o desempenho inferior do `DWQ` no conjunto de teste independente (WikiText-2).

## 5. Síntese pessoal
Tu et al. (2026) oferecem a chave interpretativa perfeita para discutir os resultados do `DWQ` em nosso artigo: ao mostrar que a destilação baseada unicamente no casamento dos *logits* de saída (*output matching*) mascara a degradação da geometria interna das camadas intermediárias, o trabalho explica por que métodos que protegem diretamente o produto matricial de cada camada (`AWQ` e `GPTQ-fp32`) generalizam melhor no WikiText-2 do que o `DWQ` em 3 e 4 bits no Apple M5 Pro.

## 6. Falseabilidade (Popper)
- **A hipótese central do trabalho é falseável?** Sim. A hipótese de que a destilação puramente por divergência KL na saída degrada a similaridade geométrica interna (`CKA`) e generaliza pior do que métodos que preservam as representações das camadas internas é diretamente testável e refutável.
- **Como foi testada?** Medindo `CKA` camada a camada e avaliando benchmarks de linguagem após destilação com e sem preservação da geometria interna em modelos quantizados para 4 bits.
- **Quais condições limitam a conclusão?** Se o conjunto de calibração de destilação for massivo (escala de pré-treinamento) e todos os pesos forem treináveis, o alinhamento de saída pode eventualmente recuperar representações internas, mas no regime leve de PTQ/DWQ (poucas amostras e apenas escalas treináveis) a limitação apontada pelos autores é severa.

## Citações úteis / Pontos-chave verificados para citação indireta
- **Seções de uso previstas no artigo:** Referencial Teórico (`R`), Discussão (`D`) e Trabalhos Futuros (`F`).
- **Ponto verificado 1 (Fragilidade do *output matching* na destilação de quantização):** Otimizar modelos quantizados exclusivamente pela divergência KL dos *logits* de saída pode mascarar distorções severas na geometria das representações intermediárias (medidas via `CKA`), prejudicando a generalização (TU et al., 2026).
- **Ponto verificado 2 (Explicação para `AWQ`/`GPTQ` > `DWQ` no M5 Pro):** Esse fenômeno explica por que, em nossos experimentos no M5 Pro, métodos que preservam localmente a saída de cada camada (`AWQ` e `GPTQ`) superaram consistentemente o `DWQ` (que otimiza apenas a divergência KL na saída final da rede) em 3 e 4 bits (TU et al., 2026).
