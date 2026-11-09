# Ficha de leitura — #26 MLX Team (2026)

**Referência (ABNT):** MLX TEAM. **Learned Quantization (DWQ, AWQ, dynamic quantization, GPTQ) — documentação do mlx-lm**. GitHub: `ml-explore/mlx-lm`, arquivo `mlx_lm/LEARNED_QUANTS.md`, 2026. Disponível em: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md. Acesso em: 22 out. 2026.
**Chave BibTeX:** `mlxlm2026dwq`
**Data da leitura:** 2026-11-09
**Escopo da leitura:** Consolidação da leitura direta da documentação técnica oficial em repositório (`mlx_lm/LEARNED_QUANTS.md` e código-fonte correspondente em `mlx_lm/quant/` na versão `0.32.2` utilizada nos experimentos), sem fabricação de números de página (documento Markdown de repositório sem paginação impressa).

## 1. Análise textual (visão global)
- **Tipo de texto:** Documentação técnica primária de engenharia de software e algoritmos do repositório oficial `ml-explore/mlx-lm` (Apple Machine Learning Research, licença MIT).
- **Estrutura conceitual:** Apresentação dos quatro métodos de quantização pós-treino aprendida/calibrada suportados pelo `mlx-lm` (`DWQ` — *Distilled Weight Quantization*, `AWQ` — *Activation-aware Weight Quantization*, *Dynamic Quantization* por sensibilidade e `GPTQ`), detalhando a interface de linha de comando, hiperparâmetros de calibração, formato de quantização afim (`mx.quantize`) e regime de uso recomendado (especialmente 2 a 4 bits).
- **Vocabulário-chave:** *Distilled Weight Quantization (DWQ)*, *non-quantized parameters*, *scales and biases*, *teacher-student KL distillation*, *mlx.core.quantize*, *group size*, *calibration_v5*.

## 2. Análise temática
- **Tema:** Especificação algorítmica e operacional dos métodos de quantização aprendida (`DWQ`, `AWQ`, `GPTQ` e quantização dinâmica) implementados nativamente para Apple Silicon na biblioteca `mlx-lm`.
- **Problema:** A quantização pós-treino por arredondamento simples (`RTN` via `mlx_lm.convert -q`) degrada acentuadamente a qualidade de modelos de linguagem em regimes de 2 a 4 bits; torna-se necessário oferecer métodos calibrados executáveis diretamente em Apple Silicon que preservem o mesmo formato final de inferência (`QuantizedLinear` com pesos inteiros empacotados, escalas e vieses por grupo) sem exigir kernels customizados em tempo de execução.
- **Tese / hipótese do(s) autor(es):** No método `DWQ` (*Distilled Weight Quantization*), manter congelados os pesos inteiros quantizados e otimizar por gradiente apenas os parâmetros em ponto flutuante não quantizados (as escalas `scales` e vieses `biases` dos grupos de quantização, além de camadas não quantizadas), utilizando o modelo original em precisão total como professor via divergência KL sobre um conjunto de calibração, melhora a retenção de qualidade em 2–4 bits mantendo 100% de compatibilidade com os kernels rápidos de inferência do `MLX`.
- **Argumentação / método:** A documentação descreve e exemplifica cada algoritmo implementado em `mlx_lm/quant/`:
  1. `DWQ` (`mlx_lm.dwq`): carrega o modelo professor em precisão original e o modelo aluno quantizado, congela os pesos inteiros (`w_q`) e executa otimização (Adam) sobre `scales` e `biases` minimizando a divergência KL entre as distribuições de saída sobre amostras do conjunto de calibração (`calibration_v5`).
  2. `AWQ` (`mlx_lm.awq`): busca em grade os fatores de escala por canal guiados pela magnitude das ativações de calibração para proteger pesos salientes antes da quantização afim.
  3. `GPTQ` (`mlx_lm.gptq`): reconstrói camada a camada os pesos usando a inversa da matriz Hessiana $H = X X^\top$ das entradas da camada acumulada sobre os lotes de calibração.
- **Resultados principais:** A documentação reporta ganhos de perplexidade dos métodos calibrados (`DWQ`, `AWQ`, `GPTQ`) sobre a conversão `RTN` padrão em 2–4 bits, sem alterar a velocidade de decodificação em inferência por gerarem tensores no mesmo esquema `mx.quantize`.

## 3. Análise interpretativa
- **Contexto e relevância:** Como o algoritmo `DWQ` foi introduzido diretamente em código pela equipe do `MLX` sem um artigo científico dedicado no arXiv ou em conferência, este documento técnico (`LEARNED_QUANTS.md`) constitui a **fonte primária indispensável** para descrever e auditar os métodos avaliados em nosso artigo.
- **Pressupostos:** A documentação assume que o hardware Apple Silicon do usuário possui memória unificada suficiente para manter o professor e o aluno simultaneamente durante o `DWQ`, e assume implicitamente que a acumulação da Hessiana no `GPTQ` no tipo de dados padrão do modelo (`bfloat16`) é numericamente estável.
- **Pontos fortes:** O design arquitetural do `mlx-lm` unifica o formato de saída de `RTN`, `GPTQ`, `AWQ` e `DWQ` no mesmo primitivo `nn.QuantizedLinear`, o que permite em nosso experimento um controle experimental perfeito: isolar o efeito do algoritmo de calibração sobre a qualidade ($Q$) e verificar que a velocidade de inferência (H4) depende apenas do bit-width e do tamanho de grupo ($g=64$).
- **Limites:** A inspeção direta da documentação e da implementação (`mlx-lm 0.32.2`) revelou três limitações críticas documentadas e resolvidas em nosso estudo:
  1. O `GPTQ` padrão acumula $H = X X^\top$ em `bfloat16` quando o modelo está em `bf16`, causando perda de positividade definida e falha da decomposição de Cholesky (`NaN`) em todos os modelos de 3B e 4B.
  2. O `GPTQ` padrão restringe os bits permitidos a $\{2, 4, 8\}$, rejeitando $3\text{ bits}$ apesar de o kernel `mx.quantize` suportar 3 bits nativamente.
  3. O `DWQ` exige manter professor + aluno + estados do otimizador simultaneamente na memória unificada, excedendo o teto operacional seguro de $18\text{ GiB}$ nos modelos de 3B e 4B em uma máquina de $24\text{ GiB}$.

## 4. Problematização
Questões e conexões diretas que o texto suscita para o nosso estudo experimental no Apple M5 Pro (24 GiB):
- **Auditoria empírica e numérica dos 4 métodos do `mlx-lm`:** Nosso artigo realiza o primeiro confronto sistemático independente entre `RTN`, `GPTQ`, `AWQ` e `DWQ` do `mlx-lm 0.32.2` em 5 modelos (0,6B a 4B) no M5 Pro:
  - Demonstramos que, embora `DWQ` supere o `RTN` em 3 e 4 bits nos modelos de 0,6B e 1,7B (reduzindo a perda $L_3 = 1 - Q_3$ em mais de 20% na H3: de $0{,}6632 \to 0{,}4572$ no `Qwen3-0.6B` e de $0{,}5921 \to 0{,}3562$ no `Qwen3-1.7B`), ele é **superado pelo `AWQ` e pelo `GPTQ-fp32`** tanto em qualidade ($Q$) quanto em pegada de memória durante a quantização (pois `AWQ` e `GPTQ` operam camada a camada/bloco a bloco e rodam com folga nos modelos de 3B e 4B dentro de 24 GiB, enquanto `DWQ` falha por memória).
- **Diagnóstico e correção do `GPTQ` (`gptq_camada.py` / `gptq_fp32`):** A leitura conjunta de `LEARNED_QUANTS.md` e do código de `mlx_lm/quant/gptq.py` permitiu identificar exatamente onde a implementação oficial diverge de Frantar et al. (2023, #08): a ausência de conversão para `float32` antes de acumular $H += X^\top X$ e aplicar o amortecimento diagonal $\lambda = 0{,}01 \cdot \text{mean}(\text{diag}(H))$. Em `bfloat16` (mantissa de 7 bits), somar $1\%$ da média diagonal a entradas diagonais maiores resulta em *no-op* por arredondamento de ponto flutuante, levando `mx.linalg.cholesky` a produzir `NaN`. Nosso método `gptq_fp32` na Etapa 4b corrige essa falha e habilita 3 bits via empacotamento bit a bit compatível com `mx.quantize`.

## 5. Síntese pessoal
A documentação `LEARNED_QUANTS.md` do `mlx-lm` (2026) é a referência técnica primária que define o funcionamento do `DWQ` e das implementações de `AWQ` e `GPTQ` no ecossistema Apple Silicon. O confronto entre essa documentação e nossos testes severos no Apple M5 Pro (24 GiB) produziu duas das principais contribuições técnicas do nosso artigo: (i) a demonstração de que `AWQ` e `GPTQ-fp32` dominam o `DWQ` em qualidade, tempo de calibração e viabilidade de memória para modelos de 3–4B; e (ii) a prova algébrica e experimental de que a acumulação da Hessiana em `float32` elimina os `NaN` do `GPTQ` padrão do `mlx-lm`.

## 6. Falseabilidade (Popper)
- **A hipótese central do documento técnico é falseável?** Sim. A afirmação operacional de que `DWQ`, `AWQ` e `GPTQ` do `mlx-lm` reduzem a perda de perplexidade em relação ao `RTN` em 2–4 bits sobre hardware Apple Silicon é diretamente testável e refutável por execução controlada.
- **Como foi testada em nosso estudo?** Executando a grade fatorial completa (5 modelos × 4 métodos × 4 larguras de bits) no M5 Pro (24 GiB), o que corroborou o ganho sobre `RTN` em 3 e 4 bits para `AWQ`, `DWQ` (em 0,6B–1,7B) e `GPTQ` (onde convergiu), mas **refutou** a estabilidade numérica do `GPTQ` padrão em `bf16` nos modelos de 3–4B (`NaN`) e a viabilidade de memória do `DWQ` em 3–4B sob 24 GiB.
- **Quais condições limitam a conclusão?** Os achados referem-se à versão `mlx-lm 0.32.2` com tamanho de grupo $g = 64$ e conjunto de calibração `calibration_v5`.

## Citações úteis / Pontos-chave verificados para citação indireta
- **Seções de uso previstas no artigo:** Referencial Teórico (`R`) e Metodologia (`M`).
- **Ponto verificado 1 (Definição operacional do DWQ):** No `mlx-lm`, o método *Distilled Weight Quantization* (`DWQ`) mantém os pesos inteiros quantizados congelados e otimiza exclusivamente os parâmetros contínuos não quantizados (escalas e vieses por grupo) para minimizar a divergência KL em relação às saídas do modelo professor em precisão original sobre o conjunto `calibration_v5` (MLX TEAM, 2026).
- **Ponto verificado 2 (Uniformidade do formato de inferência):** Todos os métodos do `mlx-lm` (`RTN`, `GPTQ`, `AWQ`, `DWQ`) convergem para a mesma representação afim empacotada por grupo (`QuantizedLinear`), garantindo que diferenças de vazão em inferência dependam apenas do bit-width e do tamanho de grupo, e não do algoritmo de calibração (MLX TEAM, 2026).
