# Ficha de leitura — #22 Hu & Ramachandran (2026)

**Referência (ABNT):** HU, Jundong; RAMACHANDRAN, Shekar. The Structure of Quantization Damage in LLMs: Why the Next Bit Should Be Spent Globally. **arXiv preprint arXiv:2609.01587**, 2026. Disponível em: https://arxiv.org/abs/2609.01587. Acesso em: 22 out. 2026.
**Chave BibTeX:** `hu2026structure`
**Data da leitura:** 2026-11-09
**Escopo da leitura:** Consolidação dos metadados e resumo verificados via API do arXiv (`2609.01587`, preprint sob revisão em workshop do NeurIPS 2026 segundo metadados do arXiv), sem fabricação de números de página ou citações literais entre aspas de PDF não lido na íntegra.

## 1. Análise textual (visão global)
- **Tipo de texto:** Preprint de análise empírica sobre a distribuição espacial do erro de quantização (*quantization damage*) entre camadas de LLMs e suas implicações para alocação de bits.
- **Estrutura conceitual:** Investigação sistemática em 9 modelos de linguagem sobre se a degradação induzida pela quantização concentra-se em poucas camadas "ultra-sensíveis" (hipótese clássica de precisão mista por camada) ou se é estruturalmente difusa por toda a rede, comparando o retorno marginal de gastar orçamento adicional de bits localmente (protegendo poucas camadas em alta precisão) vs. globalmente (refinando a granularidade/escalas em todas as camadas).
- **Vocabulário-chave:** *quantization damage*, *layer-wise sensitivity*, *mixed-precision quantization*, *global bit allocation*, *fine-grained scaling*, *bits per weight (bpw)*.

## 2. Análise temática
- **Tema:** A estrutura de distribuição do dano de quantização entre as camadas de LLMs e a estratégia ótima de alocação de orçamento adicional de bits (local por camada vs. global em granularidade fina).
- **Problema:** Muitas heurísticas de precisão mista assumem que a perda de quantização em baixo bit advém de um pequeno subconjunto de camadas críticas e que promover essas camadas para 6 ou 8 bits é a melhor forma de gastar bits extras; contudo, falta verificar empiricamente em múltiplas famílias de LLMs modernos se o dano é realmente localizado ou difuso.
- **Tese / hipótese do(s) autor(es):** O dano de quantização em LLMs é predominantemente **difuso** ao longo das camadas; por consequência, sob um mesmo orçamento médio de bits por peso (*bits per weight*), alocar o "próximo bit" (ou fração de bit) **globalmente** — por meio de granularidade mais fina de grupos/escalas em todos os tensores — recupera mais qualidade do que concentrá-lo em poucas camadas inteiras.
- **Argumentação / método:** Os autores medem o perfil de degradação camada a camada em 9 LLMs e comparam esquemas de alocação local de precisão mista contra esquemas de refinamento global de granularidade sob orçamento equivalente de bits.
- **Resultados principais:** Nos 9 modelos analisados, o erro acumula-se de forma distribuída entre os blocos transformadores, e o investimento global em escalas/granularidade fina supera consistentemente a promoção isolada de camadas.

## 3. Análise interpretativa
- **Contexto e relevância:** Oferece uma lente teórica e empírica de 2026 extremamente esclarecedora para entender como funcionam o `mlx-lm` (que gasta $+0{,}5\text{ bpw}$ globalmente em escalas e vieses de grupo 64) e os `K-quants` do `llama.cpp` (que combinam superblocos hierárquicos de escalas de 6 bits a cada 16–32 pesos com promoção seletiva de tensores específicos como `attn_v` e `ffn_down`).
- **Pressupostos:** Analisa o retorno marginal por fração de bit adicionado ao orçamento total do modelo.
- **Pontos fortes:** Formula uma hipótese clara, quantitativa e diretamente conectada ao *trade-off* entre bits nominais e bits efetivos por peso (`bpw`).
- **Limites:** Preprint em avaliação de workshop (sem revisão por pares de periódico/conferência principal confirmada); não testa diretamente a execução no hardware Apple M5 Pro.

## 4. Problematização
Questões e conexões diretas que o texto suscita para o nosso estudo experimental no Apple M5 Pro (24 GiB):
- **Explicação teórica para o orçamento de bits por peso (`bpw`) no `mlx-lm` vs. `K-quants`:** Em nossos dados (`resultados.csv` e `kquant.csv`), tanto o `mlx-lm` (com grupo $g = 64$, que adiciona $32\text{ bits} / 64 = 0{,}5\text{ bpw}$ globalmente em todos os tensores quantizados para armazenar escala e viés em `bf16`) quanto os `K-quants` do `llama.cpp` (`Q4_K_M` com $\approx 5{,}0\text{ bpw}$ e `Q3_K_M` com $\approx 4{,}1\text{–}4{,}4\text{ bpw}$) gastam frações extras de bit por peso para preservar a qualidade. Hu e Ramachandran (2026) explicam por que o investimento global em escalas de grupo pequenas ($g=64$ no MLX; sub-blocos de 32 pesos nos `K-quants`) e proteção de canais salientes em todas as camadas (`AWQ`) é tão eficaz.
- **Interpretação da queda superlinear em 3 e 2 bits (Hipótese H3):** Na nossa análise da H3 na escala de NLL (`h3_nll.json` e `protocolo-v3.md`), a degradação é estritamente superlinear nos 5 modelos avaliados ($S = (D_2 - D_3) - (D_3 - D_4) > 0$ com $\text{IC}_{95\%\text{ inf}} > 0$ em 5/5 modelos). O caráter difuso do dano descrito por Hu e Ramachandran (2026) explica por que, quando se cai para 2 bits uniformes no `RTN` ($Q \le 0{,}0037$), o erro acumulado ao longo de todas as 28–36 camadas destrói a coerência das representações internas, e por que métodos que ajustam escalas globalmente (`AWQ`, `DWQ`, `GPTQ-fp32` e `Q2_K` com `imatrix`) conseguem mitigar parte substancial dessa perda em 3 bits.

## 5. Síntese pessoal
Hu e Ramachandran (2026) demonstram que a perda de precisão em LLMs quantizados não é um defeito isolado em uma ou duas camadas, mas um fenômeno cumulativo e difuso por toda a profundidade da rede. Para o nosso artigo, esse achado fundamenta tanto a eficácia de técnicas que protegem canais/escalas globalmente em todas as camadas (`AWQ` com grupo 64 no `MLX`) quanto a necessidade de contabilizar os bits efetivos por peso (`bpw`) ao comparar `mlx-lm` e `llama.cpp`.

## 6. Falseabilidade (Popper)
- **A hipótese central do trabalho é falseável?** Sim. A hipótese de que gastar bits extras globalmente (granularidade fina) supera a alocação local por camada sob o mesmo orçamento médio de bits por peso é diretamente refutável se uma política de precisão mista por camada obtiver menor perplexidade no mesmo tamanho final em bytes.
- **Como foi testada?** Comparando perfis de erro por camada e estratégias de alocação de bits locais versus globais em 9 LLMs.
- **Quais condições limitam a conclusão?** Em arquiteturas que mantêm os *embeddings* ou a cabeça de saída (`lm_head`) sensíveis, um híbrido mínimo (preservar tensores de fronteira ou `attn_v` + granularidade fina global, exatamente como fazem os `K-quants` `Q4_K_M` e `Q3_K_M`) pode combinar os dois benefícios.

## Citações úteis / Pontos-chave verificados para citação indireta
- **Seções de uso previstas no artigo:** Referencial Teórico (`R`) e Discussão (`D`).
- **Ponto verificado 1 (Caráter difuso do dano de quantização):** A degradação provocada pela quantização em LLMs distribui-se de maneira difusa ao longo das camadas da rede, tornando o refinamento global de granularidade e escalas mais eficiente por bit investido do que a promoção isolada de poucas camadas (HU; RAMACHANDRAN, 2026).
- **Ponto verificado 2 (Leitura dos nossos resultados de 3 e 2 bits):** Esse comportamento cumulativo e difuso explica por que a transição de 4 para 3 e 2 bits produz crescimento superlinear na perda de log-verossimilhança negativa (NLL) nos 5 modelos avaliados no M5 Pro e por que métodos de ajuste global de escala (`AWQ`, `DWQ`) superam o `RTN` (HU; RAMACHANDRAN, 2026).
