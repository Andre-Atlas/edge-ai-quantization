# Ficha de leitura — 06 · Gholami et al. (2022)

**Referência (ABNT):** GHOLAMI, Amir; KIM, Sehoon; DONG, Zhen; YAO, Zhewei; MAHONEY, Michael W.; KEUTZER, Kurt. A survey of quantization methods for efficient neural network inference. *In*: **Low-Power Computer Vision**. Boca Raton: Chapman and Hall/CRC, 2022. *(Preprint disponível em: `arXiv:2103.13630`, 2021).*
**Chave BibTeX:** `gholami2022survey`
**Data da leitura:** 2026-11-09
**Escopo da leitura:** Ficha consolidada a partir dos metadados verificados via arXiv API (`2103.13630`) e da análise de resumo e estrutura técnica documentada em `literatura/mapeamento.md` e `literatura/analise-dos-artigos.md` (sem fabricação de números de página internos ou citações literais entre aspas de PDF não lido na íntegra).

## 1. Análise textual (visão global)
- **Tipo de texto:** *Survey* fundamental e capítulo de livro sobre métodos de quantização para inferência eficiente de redes neurais, elaborado pelo grupo da UC Berkeley (Kurt Keutzer, Michael W. Mahoney, Amir Gholami e colaboradores).
- **Estrutura funcional:** Apresenta rigorosamente os conceitos matemáticos e algorítmicos da quantização neural: (i) formulação básica da quantização uniforme e não-uniforme, simétrica e assimétrica (afim); (ii) granularidade dos fatores de escala (*layer-wise*, *channel-wise*, *group-wise* / *sub-channel*); (iii) *Post-Training Quantization* (PTQ) versus *Quantization-Aware Training* (QAT); (iv) métodos baseados em informação de segunda ordem (espectro da Hessiana) e alocação de precisão mista (*mixed-precision quantization*); e (v) implicações de hardware e quantização inteira pura (*integer-only*).
- **Vocabulário-chave:** *uniform/non-uniform quantization*, *symmetric/asymmetric (affine) quantization*, *scale factor*, *zero-point*, *clipping range*, *group-wise granularity*, *Hessian sensitivity*, *mixed-precision quantization*, *PTQ*, *QAT*.

## 2. Análise temática
- **Tema:** Fundamentos matemáticos, taxonomia algorítmica e implicações de hardware dos métodos de quantização de redes neurais profundas.
- **Problema:** A redução da precisão numérica de pesos e ativações de ponto flutuante (32/16 bits) para inteiros de baixa largura de bits (8, 4, 3 ou 2 bits) introduz ruído de arredondamento e truncamento (*clipping*) que degrada severamente a função de perda do modelo se a distribuição estatística dos tensores e a curvatura local do espaço de parâmetros (Hessiana) não forem respeitadas.
- **Tese / hipótese do(s) autor(es):** A eficácia da quantização depende do equilíbrio entre três mecanismos de controle do erro de quantização: o ajuste da função de mapeamento (simétrica vs. afim com ponto-zero), o refinamento da granularidade espacial das escalas (por canal ou por subgrupo de pesos) e o uso de informação de sensibilidade (como a Hessiana ou importância por camada/bloco) para proteger parâmetros de alta curvatura ou alocar larguras de bits mistas.
- **Argumentação / método:** Derivação formal dos operadores de quantização/dequantização e revisão sistemática da evolução histórica e técnica dos métodos de PTQ, QAT e precisão mista (como HAWQ), conectando a teoria de perturbação de segunda ordem (aproximação de Taylor da perda) às escolhas práticas de implementação.
- **Resultados principais:** Formaliza as equações padrão da área (como $Q(r) = \text{round}(r / S) - Z$), demonstra por que a quantização afim por grupo/canal mitiga assimetrias e *outliers* locais, e fundamenta teoricamente o uso da matriz Hessiana para medir a sensibilidade de pesos à perturbação — princípio que embasa tanto o GPTQ quanto a alocação de precisão mista dos K-quants.

## 3. Análise interpretativa
- **Contexto e pressupostos:** Considerado na literatura como o *survey* de referência canônica sobre os fundamentos de quantização neural. Os autores são também os criadores da série HAWQ (*Hessian AWare Quantization*), o que confere ao texto particular profundidade na análise de métodos de segunda ordem e precisão mista.
- **Pontos fortes:** Rigor matemático nas definições formais (simétrica vs. afim, granularidade por grupo, expansão de Taylor de segunda ordem para o erro de quantização). É a fonte primária ideal para a subseção de "Fundamentos Formais da Quantização" do nosso artigo.
- **Limites:** Como o preprint data de 2021 e o capítulo de 2022, antecede os métodos específicos de LLMs lançados entre 2023 e 2026 (GPTQ, AWQ, DWQ, K-quants); contudo, toda a maquinaria matemática que esses algoritmos utilizam já está perfeitamente formalizada em Gholami et al.

## 4. Problematização
Questões que o texto suscita para o nosso artigo Experimental no Apple M5 Pro (24 GiB de memória unificada):
- **Formalização matemática do esquema afim por grupo do MLX vs. superblocos mistos do `llama.cpp`:** Gholami et al. (2022) fornecem o vocabulário exato do nosso manuscrito. No `mlx-lm`, todos os métodos avaliados (RTN, GPTQ, AWQ, DWQ) compartilham a mesma representação física final: quantização **uniforme afim (assimétrica) por grupo** com tamanho de grupo $g=64$, onde cada bloco de 64 pesos armazena uma escala $s \in \text{fp16/bf16}$ e um viés/ponto-zero $z \in \text{fp16/bf16}$, resultando em uma sobrecarga de $32 / 64 = 0{,}5$ bit por peso (logo, 4,5 bpw em 4 bits, 3,5 bpw em 3 bits e 2,5 bpw em 2 bits). Já os K-quants do `llama.cpp` implementam a **quantização de precisão mista e escalas hierárquicas** descrita por Gholami et al., combinando sub-blocos de 32 pesos dentro de superblocos de 256 pesos e atribuindo mais bits a tensores críticos (`attn_v`, `ffn_down`).
- **Base teórica da sensibilidade de segunda ordem e diagnóstico do colapso da Hessiana `bf16`:** A formulação de Gholami et al. sobre aproximação de segunda ordem ($\Delta \mathcal{L} \approx \frac{1}{2} \Delta w^\top H \Delta w$) é a base teórica tanto para entender como o GPTQ compensa o erro de quantização quanto para diagnosticar por que a acumulação de $H = 2 X X^\top$ em precisão `bfloat16` (mantissa de apenas 7 bits) perde a positividade definida e colapsa na decomposição de Cholesky nos nossos modelos de 3B–4B parâmetros, exigindo acumulação em `float32` (`gptq_fp32`, Etapa 4b).

## 5. Síntese pessoal
Gholami et al. (2022) são a espinha dorsal matemática da seção de Referencial Teórico do nosso artigo. Todas as definições formais de operador de quantização uniforme afim, fator de escala, ponto-zero/viés, granularidade de grupo ($g=64$), sobrecarga de metadados em bits por peso (bpw) e aproximação quadrática da perda via matriz Hessiana apoiar-se-ão diretamente neste trabalho.

## 6. Falseabilidade (Popper)
- **A hipótese central do trabalho é falseável?** As proposições teóricas e empíricas sistematizadas por Gholami et al. — de que (i) a granularidade fina por grupo reduz a variância dinâmica do intervalo de quantização, (ii) a compensação de segunda ordem (Hessiana) reduz o erro quadrático de reconstrução, e (iii) a alocação de precisão mista supera a alocação uniforme sob o mesmo orçamento médio de bits — são todas rigorosamente falseáveis.
- **Como foi testada / operacionalizada em nosso estudo?**
  1. Testamos a eficácia da compensação de segunda ordem via GPTQ contra o RTN de mesmo formato físico ($g=64$) em 4 e 3 bits (`protocolo-v2` e `protocolo-v3`). Descobrimos uma condição de contorno numérica crucial: em `bf16`, a aproximação da Hessiana falha catastroficamente (`NaN`) nos modelos de 3B–4B; quando corrigida para `float32` (`gptq_fp32`), na validação em `Qwen3-0.6B` 3 bits, a perplexidade cai de $62{,}26$ (RTN 3b) para $35{,}77$ (`gptq_fp32` 3b), corroborando a teoria de segunda ordem exatamente no regime severo de 3 bits onde o erro de arredondamento do RTN explode.
  2. Testamos a precisão mista via K-quants (`Q3_K_M`, `Q2_K`) contra quantização uniforme em 3 e 2 bits, observando retenção de qualidade superior dos esquemas mistos em baixíssima precisão.
- **Quais condições limitam a conclusão?** Os métodos de segunda ordem pressupõem condicionamento numérico adequado da matriz Hessiana empírica ($H + \lambda I \succ 0$) e representatividade estatística do corpus de calibração; violações de precisão de máquina (`bf16`) invalidam a inversão de Cholesky.

## Citações úteis (curtas, com página)
- **Nota de transparência:** O PDF completo não foi lido na íntegra nesta etapa; não são incluídas citações diretas entre aspas nem números de página específicos.
- **Pontos-chave verificados para citação indireta no artigo (`\cite{gholami2022survey}`):**
  - Definições formais de quantização uniforme simétrica e assimétrica (afim), fator de escala, ponto-zero e intervalo de truncamento (Seção de Referencial Teórico).
  - Taxonomia de granularidade de quantização (*per-layer*, *per-channel*, *per-group*) e cálculo de bits efetivos por peso (Seção de Referencial Teórico e Método).
  - Fundamentação da expansão de segunda ordem baseada na Hessiana e da quantização em precisão mista (Seção de Referencial Teórico e Discussão).
