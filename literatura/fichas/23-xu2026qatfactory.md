# Ficha de leitura — #23 Xu et al. (2026)

**Referência (ABNT):** XU, Weili et al. QATFactory: A Versatile, Deployment-Aligned Framework for Quantization-aware Training and Distillation of LLMs. **arXiv preprint arXiv:2609.39223**, 2026. Disponível em: https://arxiv.org/abs/2609.39223. Acesso em: 22 out. 2026.
**Chave BibTeX:** `xu2026qatfactory`
**Data da leitura:** 2026-11-09
**Escopo da leitura:** Consolidação dos metadados e resumo verificados via API do arXiv (`2609.39223` v2, preprint sem revisão por pares localizada), sem fabricação de números de página ou citações literais entre aspas de PDF não lido na íntegra.

## 1. Análise textual (visão global)
- **Tipo de texto:** Preprint de engenharia de frameworks e algoritmos de treinamento/destilação cientes de quantização (*Quantization-Aware Training* — QAT e *Quantization-Aware Distillation* — QAD).
- **Estrutura conceitual:** Identificação do desalinhamento frequente entre a quantização simulada usada em pesquisa de QAT/QAD e os formatos reais exigidos pelos motores de produção (`vLLM` e `llama.cpp`), apresentação da plataforma unificada `QATFactory` (que incorpora QAT, QAD e aprendizado por reforço ciente de quantização — QARL), e validação com exportação direta para formatos reais incluindo `NVFP4`, `MXFP4` e `Q4_K` do `llama.cpp`.
- **Vocabulário-chave:** *Quantization-Aware Training (QAT)*, *Quantization-Aware Distillation (QAD)*, *deployment-aligned quantization*, *Q4_K*, *llama.cpp*, *NVFP4*, *MXFP4*.

## 2. Análise temática
- **Tema:** Um framework versátil e alinhado à implantação real (`QATFactory`) para treinamento e destilação cientes de quantização de LLMs, compatível com formatos de produção como `Q4_K` (`llama.cpp`), `NVFP4` e `MXFP4`.
- **Problema:** Muitos métodos de QAT e destilação para quantização (QAD) utilizam quantizadores falsos (*fake quantization*) idealizados que não correspondem à estrutura exata de blocos e escalas dos motores de inferência reais (como o formato `Q4_K` do `llama.cpp`), além de carecerem de uma base teórica/experimental unificada para destilação sobre pesos quantizados.
- **Tese / hipótese do(s) autor(es):** Alinhar o operador de quantização durante a destilação (QAD/QAT) exatamente ao formato de destino da implantação (`Q4_K`, `NVFP4`, `MXFP4`) permite recuperar a degradação de precisão em baixo bit e exportar modelos diretamente executáveis em `llama.cpp` e `vLLM`.
- **Argumentação / método:** Os autores desenvolvem o `QATFactory` integrando pipelines de QAT, QAD e QARL com suporte nativo à topologia de blocos do `Q4_K` e aos formatos de microescala de 4 bits, avaliando os modelos exportados diretamente nos motores de destino.
- **Resultados principais:** O framework demonstra ganhos consistentes de qualidade ao alinhar a destilação professor-aluno aos formatos reais de inferência, estabelecendo uma ponte direta entre QAD acadêmico e o formato `Q4_K` do `llama.cpp`.

## 3. Análise interpretativa
- **Contexto e relevância:** Cumpre um papel bibliográfico crucial em nosso artigo: como o método `DWQ` (*Distilled Weight Quantization*) do `mlx-lm` (#26) não possui artigo científico próprio publicado pela equipe da Apple, `Xu et al. (2026)` (junto com `Tu et al., 2026`, #24) fornece a fundamentação teórica contemporânea sobre a família de métodos de **Destilação Ciente de Quantização (QAD)** — na qual o modelo em precisão total atua como professor para otimizar os parâmetros do modelo quantizado minimizando a divergência entre suas distribuições de saída.
- **Pressupostos:** Assume disponibilidade de GPUs de treinamento (CUDA) para executar o laço de destilação professor-aluno antes da exportação para o motor de borda (`llama.cpp`).
- **Pontos fortes:** Conecta explicitamente a literatura de destilação para quantização (QAD) aos formatos práticos de borda (`Q4_K` do `llama.cpp`).
- **Limites:** Preprint sem revisão por pares confirmada; o laço de QAD/QAT completo é executado em clusters CUDA, não diretamente dentro de um laptop de 24 GiB.

## 4. Problematização
Questões e conexões diretas que o texto suscita para o nosso estudo experimental no Apple M5 Pro (24 GiB):
- **Fundamentação teórica do `DWQ` do `mlx-lm`:** No nosso estudo, avaliamos o `DWQ` (*Distilled Weight Quantization*, documentado em #26), que pode ser entendido como uma variante leve de QAD pós-treino: os pesos inteiros quantizados permanecem congelados e apenas os parâmetros contínuos não quantizados (escalas e vieses dos grupos de 64 pesos) são ajustados por gradiente usando a divergência KL contra os *logits* do modelo professor em `bf16`. Xu et al. (2026) situam esse paradigma dentro da família QAD alinhada ao formato de deploy.
- **Custo de memória da destilação professor-aluno (por que o `DWQ` falha em 3B–4B no M5 Pro de 24 GiB):** Enquanto Xu et al. (2026) executam QAD em servidores GPU dedicados e apenas exportam o artefato `Q4_K` para inferência no `llama.cpp`, em nosso experimento testamos a execução da própria calibração do `DWQ` localmente no Apple M5 Pro de 24 GiB. Nossos dados mostram que o `DWQ` funciona nos modelos de $0{,}6\text{B}$ e $1{,}7\text{B}$ (onde em 3 bits reduz a perda do `RTN` em $> 20\%$, atendendo à 2ª metade da H3: $L_{\text{DWQ 3b}} = 0{,}4572$ vs. $L_{\text{RTN 3b}} = 0{,}6632$ no `Qwen3-0.6B`; e $L_{\text{DWQ 3b}} = 0{,}3562$ vs. $L_{\text{RTN 3b}} = 0{,}5921$ no `Qwen3-1.7B`), mas **estoura o limite seguro de memória unificada nos modelos de 3B e 4B**, pois requer manter simultaneamente o grafo de *forward* do professor (`bf16`) e o grafo diferenciável do aluno na mesma RAM de 24 GiB.

## 5. Síntese pessoal
Xu et al. (2026) fornecem o embasamento conceitual de *Quantization-Aware Distillation* (QAD) alinhado a formatos reais de inferência (incluindo o `Q4_K` do `llama.cpp`). Em nosso artigo, esse referencial permite posicionar teoricamente o `DWQ` do `mlx-lm` como uma instância de QAD restrita às escalas/vieses e explicar por que métodos baseados em destilação professor-aluno impõem um custo de memória de calibração muito superior ao de métodos camada a camada (`GPTQ` e `AWQ`).

## 6. Falseabilidade (Popper)
- **A hipótese central do trabalho é falseável?** Sim. A afirmação de que a destilação alinhada ao formato exato de implantação (`Q4_K`, `NVFP4`, `MXFP4`) supera a quantização pós-treino sem destilação ou com quantizadores desalinhados é testável e refutável comparando a perplexidade e a acurácia do modelo exportado no runtime final.
- **Como foi testada?** Exportando modelos otimizados pelo `QATFactory` para `llama.cpp` e `vLLM` e medindo benchmarks de linguagem frente aos baselines de PTQ.
- **Quais condições limitam a conclusão?** O custo computacional e de memória da etapa de destilação é significativamente maior que o de PTQ analítica/local, restringindo onde a quantização em si pode ser executada.

## Citações úteis / Pontos-chave verificados para citação indireta
- **Seções de uso previstas no artigo:** Referencial Teórico (`R`) e Trabalhos Futuros (`F`).
- **Ponto verificado 1 (Destilação Ciente de Quantização — QAD):** A família de métodos de *Quantization-Aware Distillation* (QAD) utiliza o modelo original em precisão total como professor para guiar o ajuste dos parâmetros do modelo quantizado diretamente no formato de implantação, como o `Q4_K` do `llama.cpp` (XU et al., 2026).
- **Ponto verificado 2 (Ponte teórica para o DWQ):** Na ausência de publicação formal específica para o algoritmo `DWQ` do `mlx-lm`, a literatura de QAD fundamenta teoricamente o mecanismo de minimizar a divergência entre as distribuições do professor `bf16` e do modelo quantizado (XU et al., 2026).
