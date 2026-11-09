# Ficha de leitura — 03 · Dantas et al. (2024)

**Referência (ABNT):** DANTAS, Pierre Vilar; SABINO DA SILVA, Waldir; CORDEIRO, Lucas Carvalho et al. A comprehensive review of model compression techniques in machine learning. **Applied Intelligence**, v. 54, n. 22, 2024. DOI: `10.1007/s10489-024-05747-w`. *(Nota: lista completa de coautores e intervalo de páginas a conferir na versão final da Springer).*
**Chave BibTeX:** `dantas2024comprehensive`
**Data da leitura:** 2026-11-09
**Escopo da leitura:** Ficha consolidada a partir dos metadados verificados via Crossref (`10.1007/s10489-024-05747-w`) e da análise de resumo e estrutura técnica documentada em `literatura/mapeamento.md` e `literatura/analise-dos-artigos.md` (sem fabricação de números de página internos ou citações literais entre aspas de PDF não lido na íntegra).

## 1. Análise textual (visão global)
- **Tipo de texto:** Artigo de revisão sistemática/abrangente publicado no periódico *Applied Intelligence* (editora Springer).
- **Estrutura funcional:** Apresenta um levantamento atualizado (até 2024) das principais famílias de compressão de modelos em aprendizado de máquina, comparando poda (*pruning*), quantização (*quantization*), destilação de conhecimento (*knowledge distillation*) e decomposição tensorial/baixo posto (*low-rank factorization*) sob critérios de taxa de compressão, perda de acurácia, complexidade de implementação e compatibilidade com hardware.
- **Vocabulário-chave:** *model compression*, *pruning*, *quantization*, *knowledge distillation*, *low-rank decomposition*, *compression ratio*, *accuracy degradation*, *deployment constraints*.

## 2. Análise temática
- **Tema:** Revisão comparativa das técnicas de compressão de modelos em aprendizado de máquina e aprendizado profundo.
- **Problema:** Diante da multiplicidade de paradigmas de compressão disponíveis na literatura, engenheiros e pesquisadores enfrentam dificuldade em isolar qual família de técnica oferece a melhor relação entre redução efetiva de memória/latência e preservação da qualidade preditiva sem incorrer em custos proibitivos de retreinamento ou necessidade de hardware especializado para esparsidade.
- **Tese / hipótese do(s) autor(es):** Embora poda, destilação e fatoração alcancem altas taxas teóricas de redução de parâmetros, a quantização destaca-se pela regularidade estrutural das matrizes resultantes e pela compatibilidade imediata com unidades aritméticas inteiras e de baixa precisão do hardware moderno, podendo ser combinada (como na destilação acoplada à quantização) para mitigar perdas em regimes agressivos de compressão.
- **Argumentação / método:** Revisão comparativa da literatura recente (indexada na base Springer e bases correlatas), contrastando os fundamentos matemáticos, vantagens operacionais e limitações de implantação de cada família de compressão.
- **Resultados principais:** Sintetiza os *trade-offs* entre as abordagens: (i) poda não-estruturada gera matrizes esparsas que raramente aceleram o hardware sem suporte dedicado, enquanto poda estruturada exige ajuste fino pesado; (ii) destilação transfere conhecimento para arquiteturas menores, mas requer ciclo de treinamento; (iii) quantização reduz deterministicamente o número de bits por peso com alta previsibilidade de memória e banda.

## 3. Análise interpretativa
- **Contexto e pressupostos:** Publicado em 2024 (com autoria vinculada a grupos de pesquisa brasileiros e internacionais, como UFAM/Manchester), o artigo atualiza a literatura de compressão para o cenário contemporâneo de modelos de larga escala. Assume que a viabilidade prática de uma técnica de compressão depende da regularidade de acesso à memória no dispositivo de destino.
- **Pontos fortes:** Contraste direto e equilibrado entre as quatro grandes famílias de compressão, fornecendo argumentos sólidos para justificar o recorte metodológico de estudos empíricos focados especificamente em quantização.
- **Limites:** Por cobrir aprendizado de máquina de forma ampla (incluindo visão computacional e modelos clássicos/profundos diversos), trata os algoritmos específicos de LLMs em nível macroscópico, sendo complementado em nosso mapeamento por Zhu et al. (2024) e Gholami et al. (2022).

## 4. Problematização
Questões que o texto suscita para o nosso artigo Experimental no Apple M5 Pro (24 GiB de memória unificada):
- **Justificativa metodológica do recorte exclusivo em quantização (e exclusão de poda):** Dantas et al. (2024) sustentam diretamente a seção de Método e Delimitação do Escopo do nosso manuscrito. Em GPUs integradas de memória unificada como a do Apple M5 Pro, kernels densos de multiplicação matriz-vetor quantizada (`mlx::core::quantized_matmul` e kernels `ggml` K-quants) exploram leituras contíguas e coalizadas na memória LPDDR5X. A poda não-estruturada quebraria essa localidade espacial sem ganho real de *decode* tok/s, enquanto a quantização de 4 bits reduz exatamente pela metade (em relação a 8 bits) ou a um quarto (em relação a 16 bits) o volume de bytes trafegados por token gerado.
- **Interseção entre destilação e quantização (DWQ):** A revisão de Dantas et al. sobre combinação de paradigmas (destilação + quantização) oferece o enquadramento conceitual para o método **DWQ (*Distilled Weight Quantization*)** avaliado em nosso estudo via `mlx-lm`. Nos nossos experimentos em 4 e 3 bits, o DWQ utiliza a distribuição de probabilidades do modelo `bf16` original (*teacher*) via divergência KL para refinar as escalas e vieses não-quantizados do modelo comprimido (*student*), evitando o colapso numérico que afligiu o GPTQ `bf16` nos modelos de 3B–4B parâmetros.

## 5. Síntese pessoal
Dantas et al. (2024) cumprem dois papéis estratégicos no nosso artigo: (1) fundamentam o recorte do trabalho na quantização como técnica de primeira escolha para inferência em borda devido à regularidade de memória e ausência de necessidade de hardware de esparsidade; e (2) introduzem conceitualmente a hibridização entre quantização e destilação de conhecimento, preparando o terreno teórico para explicar o funcionamento do DWQ no ecossistema MLX e os avanços recentes de *Quantization-Aware Distillation* (QAD) discutidos na Rodada 2 (Xu et al., 2026; Tu et al., 2026).

## 6. Falseabilidade (Popper)
- **A hipótese central do trabalho é falseável?** Sim, na medida em que afirma que a quantização densa preserva a regularidade computacional e produz reduções previsíveis de memória e latência superiores às técnicas dependentes de esparsidade não-estruturada em hardware de propósito geral, e que a combinação com destilação reduz a perda de qualidade em baixas larguras de bits.
- **Como foi testada / operacionalizada em nosso estudo?** Confrontamos empiricamente a quantização pura por arredondamento (RTN), a reconstrução de segunda ordem (GPTQ), a proteção de canais salientes (AWQ) e a destilação de escalas pós-quantização (DWQ) em 5 LLMs no M5 Pro. A proposição de que a destilação pós-quantização (DWQ) melhora a retenção de qualidade sobre o RTN foi testada diretamente nas hipóteses H1 e na extensão de 3 bits (`protocolo-v3`), onde observamos ganhos consistentes em modelos como `Qwen3-1.7B` e `SmolLM3-3B`, mas margens estreitas (< 1%) em 4 bits nos modelos onde o RTN de grupo 64 já opera próximo ao teto de capacidade.
- **Quais condições limitam a conclusão?** Os ganhos da hibridização quantização+destilação dependem do tamanho do conjunto de calibração, da taxa de aprendizado das escalas e da capacidade inicial do modelo; em 2 bits extremos, nem mesmo ajustes de escala impedem a degradação severa da perplexidade.

## Citações úteis (curtas, com página)
- **Nota de transparência:** O PDF completo não foi lido na íntegra nesta etapa; não são incluídas citações diretas entre aspas nem números de página específicos (intervalo de páginas marcado como a conferir no DOI da Springer).
- **Pontos-chave verificados para citação indireta no artigo (`\cite{dantas2024comprehensive}`):**
  - Comparação sistemática entre poda, quantização, destilação de conhecimento e fatoração de baixo posto (Seção de Referencial Teórico).
  - Justificativa do recorte do estudo em quantização densa por sua compatibilidade direta com a hierarquia de memória do hardware sem exigir suporte a esparsidade (Seção de Método).
  - Enquadramento teórico da combinação entre destilação de conhecimento e quantização, base conceitual do método DWQ (Seção de Referencial Teórico e Discussão).
