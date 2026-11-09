# Quantização Pós-Treino de Pequenos Modelos de Linguagem em Memória Unificada de Borda: Um Estudo Experimental Pré-Registrado sob Critérios de Falseabilidade

**Autores:** André Atlas  
**Data da versão:** 09 de novembro de 2026  
**Repositório e Artefatos Reprodutíveis:** `https://github.com/Andre-Atlas/edge-ai-quantization` (Tags congeladas: `protocolo-v1`, `protocolo-v2`, `protocolo-v3`)

---

## Resumo

A execução local de Pequenos Modelos de Linguagem (*Small Language Models* — SLMs, de 0,6 a 4 bilhões de parâmetros) em dispositivos de borda com arquitetura de memória unificada exige equilibrar três dimensões conflitantes: retenção de qualidade preditiva, consumo de memória física e vazão de geração (*tokens*/s). Embora a literatura recente tenha caracterizado sistemas de inferência em Apple Silicon ou proposto algoritmos de quantização pós-treino (*Post-Training Quantization* — PTQ) avaliados em GPUs discretas (CUDA), permanecem escassas avaliações comparativas lado a lado que confrontem múltiplos algoritmos de PTQ na mesma máquina de memória unificada sob critérios de refutação fixados *a priori*. Fundamentado no racionalismo crítico de Karl Popper (hipóteses falseáveis e pré-registro para impedir ajustes *ad hoc*) e na metodologia estrutural de Antônio Joaquim Severino, este trabalho apresenta um estudo experimental de 93 configurações avaliadas em um MacBook Pro equipado com SoC Apple M5 Pro e 24 GiB (25,77 GB decimais) de memória unificada. Comparamos cinco algoritmos — *Round-to-Nearest* (RTN), GPTQ, *Activation-Aware Weight Quantization* (AWQ) e *Distilled Weight Quantization* (DWQ) no framework nativo MLX (`mlx-lm 0.32.2`), além de *K-quants* calibrados com matriz de importância (`imatrix`) no `llama.cpp` (`v0.7.0`) — através de quatro larguras de precisão (8, 4, 3 e 2 bits) em cinco modelos (`Qwen3-0.6B`, `Qwen3-1.7B`, `SmolLM3-3B`, `Llama-3.2-3B` e `Qwen3-4B`). As quatro hipóteses pré-registradas (H1–H4) foram submetidas a testes severos com intervalos de confiança de 95% via *bootstrap* pareado (10 000 reamostragens sobre janelas de 2048 tokens do WikiText-2) e avaliação zero-shot em cinco tarefas (`arc_easy`, `arc_challenge`, `hellaswag`, `piqa`, `winogrande`). Os resultados demonstram que: (i) **H4 é corroborada** ($E \in [0,83; 0,87] \ge 0,60$ nos modelos de 3–4B), com ganhos de decodificação de $2,93\times$ a $3,09\times$ em 4 bits frente ao modelo BF16 para uma compressão de $3,48\times$ a $3,54\times$; (ii) **H1, H2 e H3 resultam inconclusivas** sob os limiares absolutos estritos da grade original, em razão de um achado numérico central deste estudo: **a implementação padrão do GPTQ no `mlx-lm 0.32.2` produz modelos degenerados (`PPL = NaN`) em 100% dos modelos de 3 a 4 bilhões de parâmetros** (9 de 9 execuções em 8, 4 e 2 bits). Demonstramos analítica e empiricamente, por inspeção espectral camada a camada, que o colapso decorre do acúmulo do produto externo das ativações $X^\top X$ em `bfloat16` antes da soma na matriz Hessiana, gerando autovalores negativos ($\lambda_{\min}$ até $-403$) que superam o amortecimento diagonal de 1% e inviabilizam a fatoração de Cholesky nas projeções `o_proj` e `down_proj`. Propomos e validamos três contribuições de engenharia reprodutíveis: (a) `gptq_camada.py`, que reduz o pico de memória do GPTQ em $3\text{–}4\times$ com igualdade tensor a tensor bit a bit (704/704 tensores idênticos) e elimina integralmente os `NaN` quando a Hessiana é acumulada em `float32` (`gptq_fp32`, estendido a 3 bits); (b) `awq_liberar.py`, que corrige um vazamento de retenção de `input_feat` entre blocos no AWQ, reduzindo a memória ativa em $3\times$ com igualdade bit a bit; e (c) uma análise secundária pós-hoc em escala de log-verossimilhança negativa ($\Delta\text{NLL}$), que remove o efeito teto da métrica linear aos 2 bits e comprova a degradação superlinear em 5 de 5 modelos ($S_{\text{médio}} = +9,356\text{ nats/token}$, $\text{IC}_{95\%} [9,117; 9,596]$).

**Palavras-chave:** Edge AI. Quantização Pós-Treino. Pequenos Modelos de Linguagem. Memória Unificada. Apple Silicon. MLX. Falseabilidade Popperiana.

---

## Abstract

Running Small Language Models (SLMs, 0.6B–4B parameters) locally on unified-memory edge devices requires balancing predictive quality retention, physical memory footprint, and decoding throughput (tokens/s). While recent literature has either profiled system-level inference on Apple Silicon or evaluated Post-Training Quantization (PTQ) algorithms on discrete CUDA GPUs, side-by-side algorithmic comparisons on the same unified-memory hardware under pre-registered refutation criteria remain scarce. Grounded in Karl Popper's critical rationalism (falsifiable hypotheses and pre-registration against *ad hoc* adjustments) and Antônio Joaquim Severino's scientific methodology, we present an experimental study of 93 configurations on an Apple M5 Pro MacBook Pro with 24 GiB (25.77 decimal GB) of unified memory. We evaluate five PTQ approaches—Round-to-Nearest (RTN), GPTQ, Activation-Aware Weight Quantization (AWQ), and Distilled Weight Quantization (DWQ) in native MLX (`mlx-lm 0.32.2`), plus importance-matrix (`imatrix`) calibrated K-quants in `llama.cpp` (`v0.7.0`)—across 8, 4, 3, and 2 bits on five models (`Qwen3-0.6B`, `Qwen3-1.7B`, `SmolLM3-3B`, `Llama-3.2-3B`, and `Qwen3-4B`). Hypotheses H1–H4 are tested using paired bootstrap 95% confidence intervals (10,000 resamples over 2048-token WikiText-2 windows) and 5-task zero-shot accuracy. Results show that: (i) **H4 is corroborated** ($E \in [0.83, 0.87] \ge 0.60$ on 3–4B models), achieving $2.93\times\text{–}3.09\times$ faster decoding at 4 bits for a $3.48\times\text{–}3.54\times$ size reduction; (ii) **H1–H3 are inconclusive** under the strict pre-registered counts due to a primary empirical finding: **standard GPTQ in `mlx-lm 0.32.2` yields `NaN` perplexity across all 3B–4B models** (9/9 runs across 8, 4, and 2 bits). Layer-wise spectral analysis proves that accumulating $X^\top X$ in `bfloat16` induces large negative eigenvalues ($\lambda_{\min}$ down to $-403$) in `o_proj` and `down_proj` Hessians, exceeding the 1% diagonal damping and breaking Cholesky decomposition. We contribute bitwise-equivalent low-memory implementations (`gptq_camada.py` and `awq_liberar.py`, cutting peak RAM by $3\text{–}4\times$), a corrected 3/4-bit `float32`-Hessian GPTQ (`gptq_fp32`), and a post-hoc NLL-scale analysis confirming superlinear sub-4-bit degradation across all five models ($S_{\text{mean}} = +9.356\text{ nats/token}$).

**Keywords:** Edge AI. Post-Training Quantization. Small Language Models. Unified Memory. Apple Silicon. MLX. Falsifiability.

---

## 1. Introdução

A transição de Grandes Modelos de Linguagem (*Large Language Models* — LLMs) de infraestruturas de nuvem para dispositivos de borda (*Edge AI*) responde a demandas estritas de privacidade de dados, operação desconectada, previsibilidade de latência e eliminação de custos recorrentes de inferência em servidores [SHUVO et al., 2023; MENGHANI, 2023]. Nessa fronteira, consolidou-se a classe dos Pequenos Modelos de Linguagem (*Small Language Models* — SLMs, tipicamente entre 0,5 e 4 bilhões de parâmetros), projetados para caber nos limites de memória e potência térmica de computadores pessoais, estações móveis e *System-on-Chips* (SoCs) embarcados [ZHU et al., 2024; LIN et al., 2023].

Entretanto, mesmo um modelo de 4 bilhões de parâmetros representado em precisão de 16 bits (`fp16` ou `bfloat16`) ocupa cerca de 8,1 GB apenas em pesos estáticos. Durante a fase autorregressiva de geração de tokens (*decoding*), na qual um único token é produzido por passo sequencial, a inferência de transformadores é severamente limitada pela largura de banda de memória (*memory-bandwidth bound*): a unidade de processamento precisa ler repetidamente todos os pesos da memória principal para realizar poucas operações aritméticas por parâmetro [GHOLAMI et al., 2022; BENAZIR; LIN, 2026]. Além disso, em dispositivos com **arquitetura de memória unificada** (*Unified Memory Architecture* — UMA), como os processadores Apple Silicon (séries M1 a M5), CPU, GPU e subsistema operacional (como o servidor gráfico `WindowServer`) compartilham o mesmo pool físico de RAM. Nesses sistemas, picos transitórios de alocação durante a calibração, a quantização ou a avaliação de logits de vocabulário amplo podem exaurir a memória física, induzir paginação severa em disco (*swap*) ou provocar o travamento completo do sistema operacional [CHAI et al., 2025].

A **Quantização Pós-Treino** (*Post-Training Quantization* — PTQ) somente de pesos (*weight-only PTQ*) tornou-se a técnica dominante para viabilizar SLMs na borda sem o custo proibitivo de re-treinamento completo (*Quantization-Aware Training* — QAT) [NAGEL et al., 2021; DANTAS et al., 2024]. Algoritmos como *Round-to-Nearest* (RTN), GPTQ [FRANTAR et al., 2023], *Activation-Aware Weight Quantization* (AWQ) [LIN et al., 2024], *Distilled Weight Quantization* (DWQ) [MLX TEAM, 2026] e esquemas de precisão mista baseados em matrizes de importância (*K-quants* com `imatrix`) [KAWRAKOW, 2023; BORGERSEN; GOODWIN, 2025] propõem diferentes compromissos entre custo de calibração, retenção de qualidade em 4, 3 e 2 bits, e velocidade de desempacotamento em tempo de execução.

### 1.1 Problema de Pesquisa e Lacuna na Literatura

O mapeamento sistemático da literatura de 2021 a 2026 (detalhado na Seção 2) revela uma bifurcação clara. De um lado, os trabalhos seminais que introduziram algoritmos de PTQ (GPTQ, AWQ, SmoothQuant, QLoRA) e formatos sub-4-bit recentes (MXFP4, NVFP4, QATFactory) foram concebidos e avaliados quase exclusivamente em GPUs discretas NVIDIA com memória dedicada (VRAM) e kernels CUDA [FRANTAR et al., 2023; LIN et al., 2024; EGIAZARIAN et al., 2025; XU et al., 2026]. De outro lado, os estudos recentes de 2025–2026 voltados a Apple Silicon e memória unificada [BENAZIR; LIN, 2026; RAJESH et al., 2025; JAVAT; KAZAKOV, 2026; BARRIOS, 2026] concentram-se na comparação de *runtimes* de execução (MLX, `llama.cpp`, MLC-LLM, Ollama, PyTorch MPS) ou na caracterização arquitetural de sistemas de grande porte (como Mac Studio M2 Ultra de 192 GB), tratando a quantização apenas como um parâmetro fixo de configuração (tipicamente 4 bits padrão) e sem avaliar conjuntamente a degradação de qualidade e a eficiência de inferência entre diferentes famílias algorítmicas de PTQ.

Configura-se, portanto, a seguinte **lacuna científica**: *ainda são escassas avaliações experimentais comparativas que confrontem lado a lado os principais algoritmos de PTQ (RTN, GPTQ, AWQ, DWQ e K-quants) no mesmo dispositivo real de memória unificada restrita, medindo simultaneamente retenção de qualidade (perplexidade e acurácia zero-shot), pico de memória de calibração/inferência e eficiência de decodificação de 8 a 2 bits em pequenos modelos de linguagem ($\le 4\text{B}$).*

### 1.2 Fundamentação Epistemológica e Hipóteses Falseáveis

Para investigar essa lacuna com rigor metodológico, este estudo adota duplo alicerce:
1. **Diretrizes estruturais e documentais de Severino [SEVERINO, 2017]:** organização da pesquisa em etapas rastreáveis de leitura analítica (textual, temática, interpretativa, problematização e síntese), fichamento bibliográfico verificável e encadeamento lógico entre problema, objetivos e protocolo experimental.
2. **Racionalismo crítico e falseacionismo de Popper [POPPER, 1959]:** uma proposição empírica só possui caráter científico se proibir resultados observáveis e especificar, **antes** da coleta dos dados, sob quais condições experimentais será considerada **refutada**. Na prática de pesquisa em aprendizado de máquina, é comum o ajuste pós-hoc de limiares, métricas ou subconjuntos de modelos após a observação dos resultados (*degrees of freedom* do pesquisador). Para eliminar esse viés, congelamos em controle de versão (tags Git `protocolo-v1`, `protocolo-v2` e `protocolo-v3`) quatro hipóteses falseáveis (**H1–H4**) com regras matemáticas fechadas de decisão:

- **H1 (Superioridade de métodos cientes de dados em 4 bits):** *Em SLMs ($\le 4\text{B}$), a quantização de 4 bits (grupo 64) com pelo menos um algoritmo ciente de dados (GPTQ, AWQ ou DWQ) retém $\ge 95\%$ da qualidade do modelo `bfloat16` ($Q = \text{PPL}_{\text{fp}}/\text{PPL}_q \ge 0,95$), enquanto a quantização ingênua RTN não atinge esse limiar ($Q_{\text{RTN}} < 0,95$).*
- **H2 (Sensibilidade inversamente proporcional ao porte na mesma família):** *Dentro de uma mesma família arquitetural (`Qwen3`: 0,6B, 1,7B e 4B), a perda de qualidade por quantização na mesma largura de bits (4 e 3 bits) é monotônica e inversamente proporcional ao número de parâmetros ($Q_{0,6\text{B}} < Q_{1,7\text{B}} < Q_{4\text{B}}$).*
- **H3 (Regime sub-4-bit: queda superlinear e mitigação por otimização):** *Abaixo de 4 bits (transição $4 \to 3 \to 2$ bits), a degradação de qualidade no RTN cresce de forma superlinear, e métodos que otimizam parâmetros de quantização (GPTQ e DWQ) reduzem a perda relativa em 3 bits em pelo menos $20\%$ frente ao RTN ($L_{3,\text{melhor}} \le 0,80 \cdot L_{3,\text{RTN}}$).*
- **H4 (Conversão de redução de memória em vazão de decodificação):** *Em modelos de 3 a 4 bilhões de parâmetros executados em memória unificada, a redução do tamanho do modelo em 4 bits converte-se em ganho de velocidade de decodificação ($G_{\text{dec}} = \text{tok/s}_q / \text{tok/s}_{\text{fp}}$) com eficiência $E = G_{\text{dec}} / R_{\text{tam}} \ge 0,60$.*

### 1.3 Contribuições Principais

Além dos vereditos empíricos para H1–H4 sob protocolo pré-registrado, este trabalho oferece quatro contribuições técnicas e metodológicas:
1. **Descoberta e demonstração algébrica/numérica da falha de `NaN` no GPTQ do `mlx-lm 0.32.2`:** demonstramos que 100% dos modelos de 3–4B (`SmolLM3-3B`, `Llama-3.2-3B` e `Qwen3-4B`) colapsam com perplexidade `NaN` em 8, 4 e 2 bits no GPTQ padrão porque o acumulador `Catcher` calcula $X^\top X$ em `bfloat16`, gerando matrizes Hessianas indefinidas com autovalores negativos de grande magnitude ($\lambda_{\min}$ até $-403$) nas camadas `o_proj` e `down_proj`, que violam a condição de positividade definida após o amortecimento de 1% na fatoração de Cholesky.
2. **Implementações de quantização equivalentes bit a bit com redução de $3\text{–}4\times$ no pico de memória (`gptq_camada.py` e `awq_liberar.py`):** viabilizamos a calibração de modelos de 3–4B em um MacBook de 24 GiB sem alterar um único bit dos tensores gerados (704/704 tensores idênticos ao `mlx-lm` padrão nos modelos onde este não falha), além de introduzir a variante corrigida `gptq_fp32` com suporte a empacotamento de 3 bits.
3. **Caracterização empírica do pico de memória física (*footprint*) vs. memória ativa Metal/MLX:** formulamos e calibramos equações preditivas de guarda de memória que evitam o congelamento do sistema operacional em arquiteturas de memória unificada.
4. **Comparação cruzada entre ecossistemas (`mlx-lm` vs. `llama.cpp` K-quants):** avaliamos qualidade normalizada pela respectiva linha de base (`BF16`/`F16`) e vazão de prefill/decode entre formatos uniformes de grupo fixo ($g=64$) e formatos mistos de superblocos (`Q8_0`, `Q4_K_M`, `Q3_K_M`, `Q2_K` com `imatrix`).

---

## 2. Referencial Teórico e Trabalhos Relacionados

O levantamento bibliográfico foi conduzido em duas rodadas documentadas em `literatura/mapeamento.md` e nas 27 fichas analíticas em `literatura/fichas/` (01 a 27), cobrindo o período de 2021 a 2026 via APIs do Crossref, arXiv e documentação primária de repositórios.

### 2.1 Taxonomia de *Efficient Deep Learning* e Quantização Pós-Treino

Segundo a taxonomia de Menghani [2023] e as revisões de Shuvo et al. [2023], Dantas et al. [2024] e Zhu et al. [2024], a compressão de modelos para inferência eficiente divide-se em poda (*pruning*), destilação de conhecimento, fatoração de baixo posto e quantização. Em LLMs, a quantização de pesos pós-treino (*weight-only PTQ*) destaca-se por preservar a arquitetura do grafo computacional e dispensar re-treinamento caro [GHOLAMI et al., 2022; NAGEL et al., 2021].

Na quantização afim por grupos de tamanho $g$ (neste estudo, $g = 64$), um bloco de pesos contínuos $\mathbf{w} \in \mathbb{R}^g$ é mapeado para inteiros sem sinal $q_i \in \{0, 1, \dots, 2^b - 1\}$ por meio de um fator de escala $s$ e um viés (*zero-point* ou *bias*) $z$:

$$
\hat{w}_i = s \cdot q_i + z, \quad \text{com } q_i = \operatorname{clip}\!\left(\operatorname{round}\!\left(\frac{w_i - z}{s}\right), 0, 2^b - 1\right)
$$

Os cinco métodos investigados neste artigo diferem fundamentalmente em como determinam $(s, z, \mathbf{q})$:
1. **RTN (*Round-to-Nearest*):** calcula $s$ e $z$ diretamente a partir do mínimo e máximo de cada grupo de 64 pesos, sem utilizar dados de calibração. É extremamente rápido e determinístico, mas vulnerável a *outliers* e incapaz de compensar o erro introduzido nas ativações [NAGEL et al., 2021].
2. **GPTQ [FRANTAR et al., 2023]:** aproxima o problema de reconstrução camada a camada $\arg\min_{\hat{\mathbf{W}}} \|\mathbf{W}\mathbf{X} - \hat{\mathbf{W}}\mathbf{X}\|_2^2$ utilizando informação de segunda ordem derivada de *Optimal Brain Quantization* (OBQ). A matriz Hessiana da perda quadrática em relação aos pesos de uma camada linear com entradas de calibração $\mathbf{X} \in \mathbb{R}^{d_{\text{in}} \times N}$ é dada por $\mathbf{H} = 2\mathbf{X}\mathbf{X}^\top$. O algoritmo aplica amortecimento diagonal $\mathbf{H}_{\lambda} = \mathbf{H} + \lambda \mathbf{I}$ (com $\lambda = 0,01 \cdot \frac{1}{d_{\text{in}}}\operatorname{tr}(\mathbf{H})$), calcula a inversa via decomposição de Cholesky ($\mathbf{H}_{\lambda} = \mathbf{L}\mathbf{L}^\top$) e atualiza iterativamente os pesos restantes da linha para compensar o erro de quantização de cada coluna.
3. **AWQ (*Activation-Aware Weight Quantization*) [LIN et al., 2024]:** observa que uma fração diminuta ($0,1\%\text{–}1\%$) dos canais de peso processa ativações de grande magnitude (*canais salientes*). Em vez de atualizar pesos com a inversa da Hessiana (o que pode sobreajustar ao conjunto de calibração), o AWQ busca por grade um vetor de escalas por canal $\mathbf{s}$ que equilibra a proteção dos canais salientes e o erro dos canais comuns ($\mathbf{W}' = \mathbf{W}\cdot\operatorname{diag}(\mathbf{s})$, absorvendo $\mathbf{s}^{-1}$ na camada precedente), combinado ao recorte ótimo de extremos (*weight clipping*).
4. **DWQ (*Distilled Weight Quantization*) [MLX TEAM, 2026]:** congela os pesos inteiros quantizados $\mathbf{q}$ e otimiza via gradiente descendente (Adam) apenas os parâmetros contínuos de escala $s$ e viés $z$ (em `bfloat16`/`float32`), minimizando a divergência de Kullback-Leibler (KL) entre a distribuição de probabilidade de saída do modelo original (*professor*) e a do modelo quantizado (*aluno*) sobre o corpus de calibração. Fundamenta-se na família de *Quantization-Aware Distillation* (QAD) discutida em Xu et al. [2026] e Tu et al. [2026].
5. **K-quants com `imatrix` [KAWRAKOW, 2023; BORGERSEN; GOODWIN, 2025]:** implementados no ecossistema `llama.cpp`/GGUF, organizam os pesos em superblocos de 256 elementos divididos em sub-blocos de 16 ou 32 pesos, com escalas em 6 bits e alocação mista de bits por tipo de tensor (por exemplo, em `Q4_K_M`, parte dos tensores críticos `attn_v` e `ffn_down` recebe 6 bits, resultando em $\approx 4,97\text{–}5,15$ bits por peso). Quando combinados à matriz de importância (`--imatrix`), o erro quadrático de quantização é ponderado pela energia média das ativações de calibração em cada canal.

*(Nota: Métodos W8A8 como SmoothQuant [XIAO et al., 2023], ajuste fino parametrizado QLoRA [DETTMERS et al., 2023], modelos nativamente ternários BitNet b1.58 [MA et al., 2024], inferência por patches MCUNetV2 [LIN et al., 2021] e formatos de microescala FP4 com suporte de hardware dedicado em GPUs Blackwell/Hopper [EGIAZARIAN et al., 2025] integram nosso referencial teórico para delimitar fronteiras, mas não compõem a grade experimental de PTQ só-pesos em Apple Silicon.)*

### 2.2 Inferência de LLMs em Apple Silicon e Memória Unificada (2025–2026)

A segunda rodada de revisão bibliográfica identificou avanços recentes na caracterização de Apple Silicon:
- **Benazir e Lin [2026]** (*ACM SIGMETRICS PER*) perfilaram latência e vazão de LLMs em Apple Silicon frente a GPUs NVIDIA, demonstrando que o custo de desempacotamento (*dequantization overhead*) na ALU e a saturação da largura de banda da memória unificada governam a velocidade de decodificação.
- **Rajesh et al. [2025]** e **Barrios [2026]** compararam os *runtimes* MLX, `llama.cpp`, MLC-LLM, Ollama e PyTorch MPS em estações M2 Ultra e M4 Max, evidenciando a eficiência dos kernels Metal nativos do MLX e do `llama.cpp`, porém sem confrontar algoritmos de PTQ entre si.
- **Javat e Kazakov [2026]** analisaram a barreira de VRAM (*VRAM wall*) em GPUs de consumo frente à memória unificada, enquanto **Chai et al. [2025]** (*FlexQuant*) modelaram a natureza compartilhada e flutuante da memória disponível em dispositivos de borda.
- **Bergach [2026]** investigou a quantização `int4` do *KV-cache* em Apple Silicon; **Hu e Ramachandran [2026]** mostraram que o dano de quantização é difuso entre camadas; e **Taherin et al. [2026]** (*Hydra*) caracterizaram as fases de *prefill* e *decode* em SoCs NVIDIA Jetson.

A síntese desses 27 trabalhos confirma nossa tese contextual: **nenhum dos estudos mapeados compara sistematicamente RTN, GPTQ, AWQ, DWQ e K-quants no mesmo hardware Apple Silicon de memória unificada, integrando qualidade, memória de calibração/inferência e velocidade sob pré-registro experimental.**

---

## 3. Metodologia Experimental

### 3.1 Pré-Registro e Controle de Versões Metodológicas (Popper)

Para assegurar a severidade dos testes e impedir ajustes retrospectivos [POPPER, 1959], o desenho experimental seguiu um protocolo estrito de versionamento no Git:
1. **`protocolo-v1` (congelado em 22/10/2026, tag `protocolo-v1`, antes da execução da matriz):** fixou os cinco modelos, os quatro métodos MLX + K-quants, as quatro larguras de bits (8, 4, 3, 2), a métrica primária $Q = \text{PPL}_{\text{fp}} / \text{PPL}_q$, a métrica secundária $Q_{\text{acc}}$ nas cinco tarefas zero-shot e os limiares numéricos exatos de corroboração e refutação de H1–H4. Declarou explicitamente o único piloto visto antes da tag (`Qwen3-0.6B` a 4 bits).
2. **`protocolo-v2` (congelado em 25/10/2026, tag `protocolo-v2`, vigente para o estudo principal):** documentou, sem alterar nenhuma hipótese nem limiar da v1, as regras formais de tratamento de dados ausentes por limitação de ferramenta (`GPTQ` não suporta 3 bits no `mlx-lm`; `AWQ` não implementa a arquitetura `smollm3`) ou por limite físico da RAM de 24 GiB (`DWQ` em modelos de 3–4B), o tratamento do `SmolLM3-3B` como análise de sensibilidade separada (por não possuir nenhum método calibrado válido na v2), a validação de equivalência bit a bit das implementações de menor memória (`gptq_camada.py` e `awq_liberar.py`) e a inclusão declarada pós-hoc da análise de H3 em escala logarítmica ($\Delta\text{NLL}$).
3. **`protocolo-v3` (congelado em 26/10/2026, tag `protocolo-v3`):** pré-registrou a extensão experimental com a implementação `gptq_fp32` (Hessiana calculada em `float32` e empacotamento próprio de 3 bits validado contra `mx.quantize`), preservando intactos os resultados e vereditos da v2.

### 3.2 Ambiente de Hardware e Software

Todos os experimentos foram conduzidos em uma única máquina dedicada:
- **Dispositivo:** MacBook Pro com SoC **Apple M5 Pro**, **24 GiB de memória unificada** (`hw.memsize` = 25 769 803 776 bytes = **25,77 GB decimais**), executando macOS 27.0.1.
- **Pilha de Software Reprodutível (`uv.lock`):** Python `3.14.9`, `mlx 0.32.4`, `mlx-lm 0.32.2`, `numpy 2.5.4`, `huggingface_hub 2.2.0` e `lm-eval`.
- **Motor GGUF (`llama.cpp`):** versão `v0.7.0` (commit `620d222`), compilado nativamente com backend Metal (`-DGGML_METAL=ON`) via toolchain do Xcode (`xcrun`).
- **Convenção de Unidades:** todas as grandezas de memória e armazenamento são reportadas em **gigabytes decimais ($\text{GB} = 10^9\text{ bytes}$)**, mantendo consistência direta entre `psutil`, `mx.get_peak_memory()` e `/usr/bin/time -l`.

### 3.3 Modelos Avaliados e Grade Experimental

A grade principal (v2) abrange cinco modelos *base* (não ajustados por instrução, para evitar viés de *chat templates*) cobrindo duas famílias e quatro portes entre 0,6B e 4B de parâmetros, sumarizados na **Tabela 1**.

| Modelo | Camadas ($n$) | $d_{\text{model}}$ | $d_{\text{ff}}$ | $d_{\text{proj}}$ | Tam. BF16 (GB) | PPL WikiText-2 (MLX) | Acc Zero-Shot Média | Decode BF16 MLX (tok/s) | PPL F16 (`llama.cpp`) | Decode F16 (`llama.cpp`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen3-0.6B` | 28 | 1024 | 3072 | 2048 | 1.204 | 20.97 | 0.5215 | 196.3 | 18.39 | 179.1 |
| `Qwen3-1.7B` | 28 | 2048 | 6144 | 2048 | 3.453 | 16.72 | 0.6133 | 78.6 | 14.90 | 77.4 |
| `SmolLM3-3B` | 36 | 2048 | 11008 | 2048 | 6.167 | 9.12 | 0.7057 | 45.3 | 8.11 | 44.7 |
| `Llama-3.2-3B` | 28 | 3072 | 8192 | 3072 | 6.443 | 7.82 | 0.6795 | 43.0 | 6.99 | 43.8 |
| `Qwen3-4B` | 36 | 2560 | 9728 | 4096 | 8.056 | 13.66 | 0.6837 | 33.8 | 11.98 | 34.9 |

*Fonte: Gerada automaticamente por [`experimentos/gerar_tabelas_artigo.py`](file:///Users/andreatlas/Downloads/Edge%20AI%20&%20Quantization/experimentos/gerar_tabelas_artigo.py) a partir de `resultados.csv`, `kquant.csv` e `kquant-velocidade.csv`.*

### 3.4 Separação Estrita entre Fases de Qualidade e Velocidade

Um requisito metodológico crítico em hardware de memória unificada é o isolamento entre medições de qualidade e medições de tempo/vazão:
1. **Fase de Qualidade (`avaliar.py --fase qualidade`):**
   - **Perplexidade ($\text{PPL}$):** calculada sobre a partição de teste bruta do `WikiText-2` (`wikitext-2-raw-v1`), dividida em $J$ janelas contíguas de $2048$ tokens sem sobreposição ($J = 146$ nos modelos `Qwen3`, $141$ no `SmolLM3-3B` e $140$ no `Llama-3.2-3B`). Para evitar o esgotamento da memória da GPU (visto que os logits em `float32` de um único lote de 4 janelas para vocabulário de 152 mil tokens consomem $> 12\text{ GB}$, incidente registrado em 22/10), a avaliação processa estritamente **uma janela por vez**, salvando o vetor de log-verossimilhança negativa média por janela $\mathbf{nll} \in \mathbb{R}^J$ em `resultados/nll/<id>.npy`.
   - **Acurácia Zero-Shot ($\text{Acc}$):** média aritmética simples nas cinco tarefas padrão do `lm-evaluation-harness` (`arc_easy`, `arc_challenge`, `hellaswag`, `piqa`, `winogrande`), com 0-shot, lote 4 e sem template de chat.
   - **Bootstrap Pareado:** como o modelo base (`fp`) e o modelo quantizado ($q$) de uma mesma arquitetura compartilham o mesmo tokenizador, as $J$ janelas são exatamente as mesmas. A razão de qualidade $Q = \text{PPL}_{\text{fp}} / \text{PPL}_q = \exp(\overline{\text{nll}}_{\text{fp}} - \overline{\text{nll}}_q)$ tem seu intervalo de confiança de 95% ($\text{IC}_{95\%}$) estimado por $B = 10\,000$ reamostragens com reposição dos mesmos índices de janela (semente fixa 0).
2. **Fase de Velocidade (`avaliar.py --fase velocidade` e `kquant_velocidade.py`):**
   - Executada de forma **estritamente sequencial e isolada**, com a máquina ociosa, medindo um prompt fixo de $512$ tokens (*prefill*) e a geração de $128$ tokens (*decode*), com 1 rodada de aquecimento seguida de 5 repetições cronometradas.
   - Para eliminar qualquer dúvida sobre interferência de processos de usuário, a fase de velocidade completa (59 configurações MLX + 25 configurações `llama.cpp`) foi executada duas vezes em 26/10/2026: **Rodada 1** (com o aplicativo Claude Desktop aberto, monitorando controles de deriva antes/depois) e **Rodada 2 de referência** (disparada via job independente do `launchd` com o aplicativo fechado e 13,7 GB livres). A diferença média de *decode* nas 59 configurações MLX entre as duas rodadas foi de apenas $-0,4\%$ (mediana $-0,1\%$), comprovando a robustez das medidas.

### 3.5 Engenharia de Memória em Arquitetura Unificada: `gptq_camada.py` e `awq_liberar.py`

Durante a execução da Etapa 4, medições diretas do *peak memory footprint* do processo (via `/usr/bin/time -l`, que captura tanto buffers ativos do Metal quanto o cache de buffers e alocações da CPU) revelaram dois gargalos severos no `mlx-lm 0.32.2`:
1. **Acúmulo simultâneo de todas as Hessianas no GPTQ padrão:** o módulo `mlx_lm.quant.gptq` registra capturadores (`Catcher`) em todas as camadas lineares simultaneamente antes de iniciar a quantização, retendo na RAM todas as matrizes $\mathbf{H} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$ de todas as camadas ao mesmo tempo ($14,66\text{ GB}$ no `Qwen3-1.7B` e $26,06\text{ GB}$ de *footprint* no `Llama-3.2-3B`, inviabilizando o `Qwen3-4B` cuja estimativa atingia $57\text{ GB}$). Desenvolvemos o [`gptq_camada.py`](file:///Users/andreatlas/Downloads/Edge%20AI%20&%20Quantization/experimentos/gptq_camada.py), que propaga as ativações do modelo original camada por camada, acumulando, invertendo e liberando a Hessiana de uma única camada por vez. A validação contra o `mlx-lm` padrão em `Qwen3-0.6B` e `Qwen3-1.7B` comprovou **704 de 704 tensores idênticos bit a bit** ($\max|\Delta\text{NLL}| = 0$), reduzindo o pico de memória MLX de $5,80 \to 1,44\text{ GB}$ (0,6B) e de $14,66 \to 3,31\text{ GB}$ (1,7B).
2. **Vazamento de retenção de `input_feat` entre blocos no AWQ padrão:** o perfilamento com [`perfil_awq.py`](file:///Users/andreatlas/Downloads/Edge%20AI%20&%20Quantization/experimentos/perfil_awq.py) demonstrou que a memória ativa no AWQ crescia linearmente $\approx 1,34\text{ GB}$ por camada porque os tensores `module.input_feat` permaneciam retidos em referências cíclicas do Python e no cache ilimitado do Metal ($24,20\text{ GB}$ no `Qwen3-1.7B`). Desenvolvemos o [`awq_liberar.py`](file:///Users/andreatlas/Downloads/Edge%20AI%20&%20Quantization/experimentos/awq_liberar.py), que invoca `gc.collect()` antes de `mx.clear_cache()` ao final de cada bloco e limita o cache do Metal a 1 GB. A validação confirmou **704 de 704 tensores idênticos bit a bit**, reduzindo o pico MLX de $13,42 \to 4,17\text{ GB}$ (0,6B) e de $24,20 \to 8,44\text{ GB}$ (1,7B), o que permitiu executar toda a família AWQ (8, 4, 3 e 2 bits) no `Llama-3.2-3B` e no `Qwen3-4B` com *footprint* estável entre $12,27$ e $14,99\text{ GB}$.
3. **Modelagem preditiva do guarda de memória (`matriz.py`):** com base nas 10 medições empíricas de *footprint* do AWQ, substituímos a extrapolação linear em função do tamanho dos pesos por um modelo físico baseado no volume das entradas de um bloco $\text{input\_feat}_{\text{GB}} = 128 \times 512 \times 2\text{ B} \times (3d + o_{\text{in}} + 2d + i) \times 10^{-9}$ e na largura de bits $b$:

$$
\text{Footprint}_{\text{AWQ}}(\text{GB}) = -1,13 + 3,66 \cdot \text{input\_feat}_{\text{GB}} + 0,40 \cdot b + 0,40\text{ (margem de segurança)}
$$

com resíduo máximo de apenas $0,23\text{ GB}$ e zero subestimações nas 10 configurações medidas.

---

## 4. Resultados e Discussão

A execução completa da Matriz v2 totalizou **68 jobs de qualidade** (59 válidos e 9 inválidos por `NaN`, além de 17 combinações excluídas declaradamente por limitação de ferramenta ou memória do DWQ em 3–4B), **25 configurações de K-quants** no `llama.cpp` e **84 medições de velocidade** na rodada de referência (59 MLX + 25 GGUF).

![Figura 1 — Retenção de Qualidade vs. Tamanho do Modelo](/Users/andreatlas/Downloads/Edge AI & Quantization/artigo/figuras/figura1_pareto_qualidade_tamanho.svg)

### 4.1 Resultados de Qualidade no MLX e Vereditos de H1 e H2

A **Tabela 2** apresenta a matriz completa de retenção de qualidade $Q = \text{PPL}_{\text{fp}}/\text{PPL}_q$ e acurácia relativa $Q_{\text{acc}} = \text{Acc}_q/\text{Acc}_{\text{fp}}$ para todos os modelos, métodos e larguras de bits avaliados no MLX.

| Modelo | Bits | RTN ($Q$ / $Q_{\text{acc}}$) | GPTQ-bf16 ($Q$ / $Q_{\text{acc}}$) | GPTQ-fp32 ($Q$ / $Q_{\text{acc}}$) | AWQ ($Q$ / $Q_{\text{acc}}$) | DWQ ($Q$ / $Q_{\text{acc}}$) |
|---|---:|---|---|---|---|---|
| `Qwen3-0.6B` | 8 | 0.9965 (0.998) | 0.9920 (1.005) | — | 0.9856 (1.000) | 0.9965 (1.001) |
| `Qwen3-0.6B` | 4 | 0.8618 (0.938) | 0.9155 (0.956) | 0.9285 (0.951) | 0.8839 (0.970) | 0.8968 (0.970) |
| `Qwen3-0.6B` | 3 | 0.3368 (0.814) | — | 0.5216 (0.837) | 0.6276 (0.874) | 0.5428 (0.904) |
| `Qwen3-0.6B` | 2 | 0.0000 (0.687) | 0.0009 (0.689) | — | 0.0000 (0.699) | 0.0022 (0.687) |
| `Qwen3-1.7B` | 8 | 0.9924 (1.002) | 1.0010 (1.000) | — | 1.0007 (0.998) | 0.9924 (1.003) |
| `Qwen3-1.7B` | 4 | 0.8555 (0.968) | 0.9194 (0.991) | — | 0.9702 (1.000) | 0.9336 (0.993) |
| `Qwen3-1.7B` | 3 | 0.4079 (0.814) | — | — | 0.7903 (0.888) | 0.6438 (0.910) |
| `Qwen3-1.7B` | 2 | 0.0000 (0.579) | 0.0002 (0.588) | — | 0.0003 (0.617) | 0.0037 (0.595) |
| `SmolLM3-3B` | 8 | 0.9992 (0.999) | **NaN** | — | — | — |
| `SmolLM3-3B` | 4 | 0.8524 (0.979) | **NaN** | — | — | — |
| `SmolLM3-3B` | 3 | 0.0911 (0.874) | — | — | — | — |
| `SmolLM3-3B` | 2 | 0.0000 (0.500) | **NaN** | — | — | — |
| `Llama-3.2-3B` | 8 | 0.9992 (1.000) | **NaN** | — | 0.9944 (1.000) | — |
| `Llama-3.2-3B` | 4 | 0.9316 (0.975) | **NaN** | — | 0.9554 (0.988) | — |
| `Llama-3.2-3B` | 3 | 0.6394 (0.893) | — | — | 0.8090 (0.937) | — |
| `Llama-3.2-3B` | 2 | 0.0026 (0.540) | **NaN** | — | 0.0162 (0.666) | — |
| `Qwen3-4B` | 8 | 1.0055 (1.000) | **NaN** | — | 0.9897 (1.000) | — |
| `Qwen3-4B` | 4 | 0.9418 (0.970) | **NaN** | — | 0.9317 (0.967) | — |
| `Qwen3-4B` | 3 | 0.6523 (0.873) | — | — | 0.8069 (0.901) | — |
| `Qwen3-4B` | 2 | 0.0012 (0.530) | **NaN** | — | 0.1703 (0.652) | — |

#### Análise de H1 (4 bits: métodos cientes $\ge 0,95$ vs. RTN $< 0,95$)
Nos quatro modelos avaliáveis da grade v2 (`Qwen3-0.6B`, `Qwen3-1.7B`, `Llama-3.2-3B` e `Qwen3-4B`; o `SmolLM3-3B` não possui método ciente válido na v2), **em nenhum modelo o RTN atingiu o limiar de 0,95** ($Q_{\text{RTN}} \in [0,8524; 0,9418]$, todos com limite superior do $\text{IC}_{95\%} < 0,95$). Isso confirma metade da intuição teórica: a quantização ingênua RTN em 4 bits impõe perda estatisticamente significativa de perplexidade (entre $5,8\%$ e $14,8\%$). Contudo, a condição $\text{IC}_{\inf}(Q_{\text{melhor ciente}}) \ge 0,95$ foi satisfeita em **2 dos 4 modelos**:
- `Qwen3-1.7B`: AWQ atinge $Q = 0,9702$ ($\text{IC}_{95\%} [0,9659; 0,9745]$), contra $0,8555$ do RTN;
- `Llama-3.2-3B`: AWQ atinge $Q = 0,9554$ ($\text{IC}_{95\%} [0,9538; 0,9571]$), contra $0,9316$ do RTN.

Nos outros dois modelos, nenhum método ciente atingiu $0,95$: no `Qwen3-0.6B`, o melhor método da v2 foi o GPTQ ($Q = 0,9155$, ou $0,9285$ com `gptq_fp32`), refletindo a maior fragilidade do modelo de 0,6B; já no `Qwen3-4B`, com o GPTQ padrão invalidado por `NaN` e o DWQ excluído por memória, o único método ciente disponível (AWQ) obteve $Q = 0,9317$ ($\text{IC}_{95\%} [0,9280; 0,9355]$), ligeiramente abaixo do próprio RTN ($0,9418$). Como o protocolo pré-registrado exigia $\ge 4$ sucessos para corroboração ou $\ge 3$ fracassos para refutação, **o veredito formal de H1 na v2 é inconclusivo**.

#### Análise de H2 (Efeito de porte na família `Qwen3`: $0,6\text{B} < 1,7\text{B} < 4\text{B}$)
No RTN a **3 bits**, a ordem monotônica confirma-se com folga estatística: $Q_{0,6\text{B}} = 0,3368$ ($\text{IC}_{95\%} [0,3320; 0,3415]$) $< Q_{1,7\text{B}} = 0,4079$ ($[0,4005; 0,4153]$) $< Q_{4\text{B}} = 0,6523$ ($[0,6433; 0,6612]$). Em **4 bits**, porém, embora o modelo de 4B ($Q = 0,9418$) supere amplamente os menores, ocorre empate técnico entre `Qwen3-0.6B` ($0,8618$) e `Qwen3-1.7B` ($0,8555$). Já nos métodos calibrados (AWQ e DWQ, avaliados suplementarmente entre 0,6B e 1,7B), a relação $Q_{0,6\text{B}} < Q_{1,7\text{B}}$ confirma-se em **4 de 4 combinações** (AWQ 4b: $0,8839 < 0,9702$; AWQ 3b: $0,6276 < 0,7903$; DWQ 4b: $0,8968 < 0,9336$; DWQ 3b: $0,5428 < 0,6438$). Com 1 de 2 células completas do RTN confirmadas ($50\%$), **H2 resulta inconclusiva** na grade v2.

### 4.2 O Achado Central: Por Que o GPTQ Padrão (`mlx-lm 0.32.2`) Colapsa com `NaN` nos Modelos de 3–4B?

Um resultado inesperado e de alto impacto prático emergiu na Etapa 4: **todas as 9 execuções do GPTQ padrão (`bf16`) nos três modelos de 3 a 4 bilhões de parâmetros (`SmolLM3-3B`, `Llama-3.2-3B` e `Qwen3-4B`, nas larguras de 8, 4 e 2 bits) produziram modelos corrompidos com `PPL = NaN` e acurácia zero-shot exatamente no nível do acaso ($0,3438$, $Q_{\text{acc}} \approx 0,50\text{–}0,59$)**, ao passo que `Qwen3-0.6B` e `Qwen3-1.7B` convergiram normalmente.

Para isolar a causa-raiz sem incorrer em especulação, conduzimos uma investigação em três passos registrada nas Fases 6, 9 e 10 do caderno de laboratório:
1. **Inspeção estrutural dos tensores salvos:** no `SmolLM3-3B`, constatamos que exatamente os mesmos 8 tensores (escalas e vieses de `self_attn.o_proj` nas camadas 0, 5 e 34, e `mlp.down_proj` na camada 10) continham `NaN`/`Inf` nas quantizações de **8, 4 e 2 bits**, com exatamente 31 de 32 grupos corrompidos por tensor. A invariância em relação ao número de bits provou que o `NaN` nasce **antes** do laço de quantização, na inversão da matriz Hessiana.
2. **Análise espectral de `Catcher` em `bfloat16` vs. `float32` (`testar_hessiana.py`):** no `mlx-lm 0.32.2`, a classe `Catcher` acumula a matriz Hessiana fazendo:
   ```python
   H += mx.matmul(x.T, x).astype(mx.float32)
   ```
   Como as ativações `x` estão em `bfloat16` (mantissa de apenas 7 bits, precisão relativa $\approx 7,8 \times 10^{-3}$), o produto interno de $65\,536$ tokens ($128 \times 512$) é computado e truncado em `bfloat16` **antes** da conversão para `float32`. Matematicamente, enquanto a matriz exata $\mathbf{H}_{\text{fp32}} = \mathbf{X}_{\text{fp32}}^\top \mathbf{X}_{\text{fp32}}$ é garantidamente semidefinida positiva ($\mathbf{v}^\top \mathbf{H}_{\text{fp32}}\mathbf{v} = \|\mathbf{X}\mathbf{v}\|_2^2 \ge 0$, logo $\lambda_{\min}(\mathbf{H}_{\text{fp32}}) \ge 0$), a matriz arredondada $\tilde{\mathbf{H}}_{\text{bf16}} = \operatorname{fl}_{\text{bf16}}(\mathbf{X}^\top \mathbf{X})$ perde a propriedade de semidefinitude positiva quando a distribuição das ativações de entrada é altamente anisotrópica (como ocorre na entrada de `o_proj` e `down_proj` de modelos maiores).
3. **Verificação empírica dos autovalores em todas as camadas:** executando `testar_hessiana.py` em todas as camadas dos três modelos de 3–4B, comprovamos que $\tilde{\mathbf{H}}_{\text{bf16}}$ apresenta autovalores mínimos fortemente negativos que superam em módulo o amortecimento diagonal $\lambda_{\text{damp}} = 0,01 \cdot \bar{h}_{ii}$:
   - **`SmolLM3-3B`:** falham `o_proj` nas camadas 0, 5 e 34 (na camada 34, $\lambda_{\min} = -103,0$ contra amortecimento $\lambda_{\text{damp}} = +25,5$) e `down_proj` na camada 10.
   - **`Llama-3.2-3B`:** falha logo na camada 0 em `o_proj` ($\lambda_{\min} = -2,42$ contra amortecimento $\lambda_{\text{damp}} = +0,41$).
   - **`Qwen3-4B`:** falham 7 matrizes (`o_proj` nas camadas 0, 1, 24, 26 e 30 — onde $\lambda_{\min} = -403,0$ contra $\lambda_{\text{damp}} = +254,0$ — e `down_proj` nas camadas 6 e 35).
   - **Com o produto acumulado em `float32` (`--hessiana fp32`):** **nenhuma matriz falha (0 de 396 matrizes nos três modelos)**, e todos os autovalores permanecem estritamente positivos após o amortecimento.

No `Qwen3-0.6B`, a avaliação do método corrigido `gptq_fp32` (Etapa 4b) já demonstrou redução adicional da perplexidade em 4 bits de $22,9027$ (`bf16`) para **$22,5806$ (`fp32`)** ($Q$ sobe de $0,9155$ para **$0,9285$**, razão $Q_{\text{fp32}}/Q_{\text{bf16}} = 1,0143$, $\text{IC}_{95\%} [1,0127; 1,0159]$), e em 3 bits produziu perplexidade finita de **$40,1984$** ($Q = 0,5216$, superando amplamente os $62,2650$ do RTN 3 bits e validando o critério da Seção 4.2 do `protocolo-v3`).

### 4.3 Regime Sub-4-Bit (H3): Efeito Teto na Escala Linear vs. Superlinearidade na Escala Logarítmica

![Figura 2 — Degradação Superlinear em Escala Logarítmica (NLL)](/Users/andreatlas/Downloads/Edge AI & Quantization/artigo/figuras/figura2_degradacao_superlinear_nll.svg)

A hipótese **H3** testa dois fenômenos: (1ª metade) se a queda de qualidade de $4 \to 3 \to 2$ bits é superlinear no RTN; e (2ª metade) se métodos baseados em otimização (GPTQ/DWQ) reduzem a perda $L_3 = 1 - Q_3$ em pelo menos $20\%$ frente ao RTN.
- **Na 2ª metade (3 bits):** onde os métodos puderam ser executados (`Qwen3-0.6B` e `Qwen3-1.7B`), a redução de perda pelo DWQ foi expressiva: no `0.6B`, $L_3$ caiu de $0,6632$ (RTN) para $0,4572$ (DWQ, redução de **$31,1\%$**, razão $0,689$); no `1.7B`, caiu de $0,5921$ para $0,3562$ (redução de **$39,9\%$**, razão $0,601$). Além disso, embora o AWQ não fizesse parte da regra original da 2ª metade da H3, ele reduziu $L_3$ em todos os modelos de 3–4B (`Llama-3.2-3B`: $Q_3$ salta de $0,6394$ no RTN para **$0,8090$** no AWQ; `Qwen3-4B`: de $0,6523$ para **$0,8069$**).
- **Na 1ª metade (superlinearidade e o artefato de saturação de $L = 1 - Q$):** nos 4 modelos contados da v2, os incrementos médios de $L$ no RTN foram $\Delta L_{4\to 3} = 0,3886$ e $\Delta L_{3\to 2} = 0,5082$ (superlinear). Porém, ao incluir o `SmolLM3-3B` na análise de sensibilidade, a média se inverte ($0,4631$ vs. $0,4247$). A inspeção matemática em `analise_h3_nll.py` revelou o motivo: como $Q = \text{PPL}_{\text{fp}}/\text{PPL}_q \ge 0$, a perda limitada $L = 1 - Q$ nunca pode ultrapassar $1,0$. No `SmolLM3-3B`, a perda em 3 bits já atinge $L_3 = 0,9089$ ($Q_3 = 0,0911$); portanto, o incremento máximo matematicamente possível entre 3 e 2 bits é limitado pelo teto $1 - L_3 = 0,0911 < \Delta L_{4\to 3} = 0,7613$, ainda que a perplexidade exploda de $100,1$ (3 bits) para $8,05 \times 10^{10}$ (2 bits)!
- **Análise na escala natural de entropia cruzada ($\Delta\text{NLL}$, Figura 2):** medindo a degradação diretamente na escala aditiva de log-verossimilhança negativa $D_b = \text{NLL}_{q,b} - \text{NLL}_{\text{fp}} = \ln(\text{PPL}_{q,b}/\text{PPL}_{\text{fp}})$ em nats/token, a segunda diferença $S = (D_2 - D_3) - (D_3 - D_4)$ é **positiva e estatisticamente significativa nos 5 de 5 modelos**:
  - `Qwen3-0.6B`: $S = +8,455\text{ nats/tok}$ ($\text{IC}_{95\%} [8,361; 8,550]$);
  - `Qwen3-1.7B`: $S = +8,930\text{ nats/tok}$ ($[8,812; 9,047]$);
  - `SmolLM3-3B`: $S = +18,271\text{ nats/tok}$ ($[17,093; 19,459]$);
  - `Llama-3.2-3B`: $S = +5,140\text{ nats/tok}$ ($[5,063; 5,215]$);
  - `Qwen3-4B`: $S = +5,972\text{ nats/tok}$ ($[5,897; 6,047]$);
  - **Média agregada (5 modelos):** $S_{\text{médio}} = +9,356\text{ nats/tok}$ ($\text{IC}_{95\%} [9,117; 9,596]$). Em conformidade com Popper, mantemos o veredito primário da v2 como **inconclusivo** e reportamos a corroboração na escala de NLL como análise secundária pós-hoc na v2 (e critério pré-registrado na v3).

### 4.4 Velocidade de Inferência, Largura de Banda e Corroboração da H4

![Figura 3 — Eficiência de Banda E em 4 Bits por Porte de Modelo (H4)](/Users/andreatlas/Downloads/Edge AI & Quantization/artigo/figuras/figura3_eficiencia_velocidade_h4.svg)

A **Tabela 3** e a **Figura 3** sintetizam os resultados de velocidade de inferência (Rodada 2 de referência, máquina ociosa) e a eficiência de conversão de compressão em aceleração de decodificação $E = G_{\text{dec}} / R_{\text{tam}}$ a 4 bits.

| Modelo | Método (4b) | Tam. (GB) | Compressão ($R_{\text{tam}}$) | Prefill (tok/s) | Decode (tok/s) | Ganho Decode ($G_{\text{dec}}$) | Eficiência $E = G_{\text{dec}}/R_{\text{tam}}$ | Pico Memória GPU (GB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen3-0.6B` | `RTN` | 0.347 | 3.47× | 14583.8 | 400.6 | 2.04× | **0.588** | 0.91 |
| `Qwen3-0.6B` | `GPTQ` | 0.386 | 3.12× | 14140.4 | 390.2 | 1.99× | **0.637** | 0.95 |
| `Qwen3-0.6B` | `AWQ` | 0.357 | 3.37× | 14393.6 | 395.7 | 2.02× | **0.598** | 0.91 |
| `Qwen3-0.6B` | `DWQ` | 0.347 | 3.47× | 14410.1 | 399.0 | 2.03× | **0.586** | 0.91 |
| `Qwen3-1.7B` | `RTN` | 0.980 | 3.52× | 6337.6 | 222.6 | 2.83× | **0.803** | 1.71 |
| `Qwen3-1.7B` | `GPTQ` | 1.057 | 3.27× | 6276.1 | 211.0 | 2.68× | **0.821** | 1.80 |
| `Qwen3-1.7B` | `AWQ` | 0.999 | 3.46× | 6322.8 | 220.8 | 2.81× | **0.812** | 1.74 |
| `Qwen3-1.7B` | `DWQ` | 0.980 | 3.52× | 6316.4 | 223.5 | 2.84× | **0.807** | 1.71 |
| `SmolLM3-3B` | `RTN` | 1.747 | 3.53× | 3077.4 | 132.6 | 2.93× | **0.829** | 2.48 |
| `Llama-3.2-3B` | `RTN` | 1.825 | 3.53× | 3303.4 | 125.8 | 2.93× | **0.830** | 2.49 |
| `Llama-3.2-3B` | `AWQ` | 1.849 | 3.48× | 3444.4 | 131.5 | 3.06× | **0.878** | 2.52 |
| `Qwen3-4B` | `RTN` | 2.275 | 3.54× | 2538.0 | 104.4 | 3.09× | **0.872** | 2.99 |
| `Qwen3-4B` | `AWQ` | 2.299 | 3.50× | 2466.6 | 103.5 | 3.06× | **0.874** | 3.02 |

Os dados **corroboram a hipótese H4 com ampla margem**: nos três modelos de 3–4B, a eficiência média por modelo é de **$0,829$ (`SmolLM3-3B`), $0,854$ (`Llama-3.2-3B`) e $0,873$ (`Qwen3-4B`)**, todas substancialmente superiores ao limiar pré-registrado de $0,60$. Observa-se ademais uma lei de escala clara em função do porte do modelo:
- No modelo menor (`Qwen3-0.6B`, $0,35\text{ GB}$ em 4 bits), $E \in [0,586; 0,637]$ ($G_{\text{dec}} \approx 2,0\times$), pois os pesos cabem com folga nos caches do SoC e uma fração não desprezível da latência por token provém do despacho de kernels Metal e das camadas de atenção/normalização.
- A partir de `1.7B` ($E \approx 0,81$, $G_{\text{dec}} \approx 2,8\times$) e `3B–4B` ($E \in [0,83; 0,88]$, $G_{\text{dec}} \approx 3,0\text{–}3,1\times$), a decodificação torna-se puramente limitada pela largura de banda da memória unificada [BENAZIR; LIN, 2026], convertendo quase linearmente a redução de bytes lidos em vazão de tokens por segundo.
- Também documentamos e tratamos uma anomalia experimental importante (Fase 20 do caderno): o modelo degenerado `Llama-3.2-3B AWQ 2b` ($Q = 0,0162$) emitia o token de fim de sequência (`EOS`) após gerar apenas **1 token**, produzindo uma velocidade aparente espúria de $\approx 49\,294\text{ tok/s}$. Modificamos `avaliar.py` para registrar `decode_tokens_min` e sinalizar automaticamente gerações truncadas por `EOS` prematuro.

### 4.5 Comparação com `llama.cpp` (K-quants com `imatrix`)

![Figura 4 — Velocidade de Decodificação em 4 Bits: MLX vs. llama.cpp](/Users/andreatlas/Downloads/Edge AI & Quantization/artigo/figuras/figura4_mlx_vs_kquants.svg)

A **Tabela 4** e a **Figura 4** apresentam o comportamento dos formatos K-quants (`Q8_0`, `Q4_K_M`, `Q3_K_M`, `Q2_K` calibrados com `imatrix` sobre o mesmo texto `calibration_v5`) no `llama.cpp`. Como o `llama-perplexity` utiliza janelamento distinto do avaliador MLX, comparamos exclusivamente as razões adimensionais contra o respectivo baseline em precisão cheia ($Q_{\text{GGUF}} = \text{PPL}_{\text{F16}}/\text{PPL}_q$).

| Modelo | Tipo | Bits/Peso (bpw) | Tam. (GB) | PPL (`llama.cpp`) | $Q_{\text{GGUF}} = \text{PPL}_{\text{F16}}/\text{PPL}_q$ | KLD Média | Top-1 Igual (%) | Prefill (tok/s) | Decode (tok/s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen3-0.6B` | `F16` | 16.07 | 1.509 | 18.39 | 1.0000 | 0.0000 | 100.0 | 13189.0 | 179.1 |
| `Qwen3-0.6B` | `Q8_0` | 8.56 | 0.805 | 18.43 | 0.9980 | 0.0011 | 98.2 | 13482.8 | 230.6 |
| `Qwen3-0.6B` | `Q4_K_M` | 5.15 | 0.484 | 19.59 | 0.9389 | 0.0672 | 87.0 | 13314.6 | 274.2 |
| `Qwen3-0.6B` | `Q3_K_M` | 4.41 | 0.414 | 21.37 | 0.8607 | 0.2600 | 76.2 | 13223.2 | 263.7 |
| `Qwen3-0.6B` | `Q2_K` | 3.70 | 0.347 | 37.03 | 0.4967 | 0.8491 | 59.1 | 13546.2 | 270.4 |
| `Qwen3-1.7B` | `F16` | 16.02 | 4.070 | 14.90 | 1.0000 | 0.0000 | 100.0 | 5494.4 | 77.4 |
| `Qwen3-1.7B` | `Q8_0` | 8.53 | 2.165 | 15.09 | 0.9874 | 0.0015 | 98.4 | 5549.0 | 129.5 |
| `Qwen3-1.7B` | `Q4_K_M` | 5.05 | 1.282 | 15.53 | 0.9597 | 0.0457 | 91.2 | 5277.5 | 184.1 |
| `Qwen3-1.7B` | `Q3_K_M` | 4.23 | 1.073 | 16.25 | 0.9170 | 0.1624 | 83.8 | 5278.9 | 177.3 |
| `Qwen3-1.7B` | `Q2_K` | 3.46 | 0.880 | 20.45 | 0.7287 | 0.5004 | 72.7 | 5430.7 | 190.2 |
| `SmolLM3-3B` | `F16` | 16.02 | 6.158 | 8.11 | 1.0000 | 0.0000 | 100.0 | 2802.4 | 44.7 |
| `SmolLM3-3B` | `Q8_0` | 8.52 | 3.276 | 8.12 | 0.9993 | 0.0008 | 98.2 | 2957.2 | 77.4 |
| `SmolLM3-3B` | `Q4_K_M` | 4.98 | 1.915 | 8.31 | 0.9762 | 0.0310 | 91.0 | 2817.2 | 119.1 |
| `SmolLM3-3B` | `Q3_K_M` | 4.09 | 1.571 | 8.67 | 0.9355 | 0.0871 | 86.0 | 2797.6 | 121.7 |
| `SmolLM3-3B` | `Q2_K` | 3.26 | 1.253 | 10.65 | 0.7619 | 0.3349 | 73.9 | 2845.8 | 138.1 |
| `Llama-3.2-3B` | `F16` | 16.02 | 6.434 | 6.99 | 1.0000 | 0.0000 | 100.0 | 2957.2 | 43.8 |
| `Llama-3.2-3B` | `Q8_0` | 8.52 | 3.422 | 6.99 | 0.9995 | 0.0003 | 99.1 | 2936.6 | 76.3 |
| `Llama-3.2-3B` | `Q4_K_M` | 5.03 | 2.019 | 7.14 | 0.9794 | 0.0224 | 92.8 | 2674.3 | 116.9 |
| `Llama-3.2-3B` | `Q3_K_M` | 4.20 | 1.687 | 7.47 | 0.9364 | 0.0693 | 87.7 | 2577.0 | 118.8 |
| `Llama-3.2-3B` | `Q2_K` | 3.40 | 1.364 | 9.83 | 0.7113 | 0.3544 | 73.9 | 2590.9 | 136.0 |
| `Qwen3-4B` | `F16` | 16.01 | 8.051 | 11.98 | 1.0000 | 0.0000 | 100.0 | 2165.6 | 34.9 |
| `Qwen3-4B` | `Q8_0` | 8.51 | 4.280 | 12.12 | 0.9879 | 0.0005 | 99.1 | 2173.9 | 60.5 |
| `Qwen3-4B` | `Q4_K_M` | 4.97 | 2.497 | 12.20 | 0.9817 | 0.0377 | 92.3 | 2005.7 | 93.8 |
| `Qwen3-4B` | `Q3_K_M` | 4.13 | 2.076 | 13.74 | 0.8714 | 0.1310 | 86.3 | 1946.0 | 89.4 |
| `Qwen3-4B` | `Q2_K` | 3.32 | 1.669 | 15.54 | 0.7706 | 0.3980 | 76.2 | 1982.3 | 106.0 |

Três contrastes técnicos destacam-se na comparação entre `mlx-lm` e `llama.cpp`:
1. **Evitação do colapso em `Q2_K` e `Q3_K_M` graças à alocação de bits mistos:** enquanto todos os métodos uniformes de 2 bits do MLX ($2,5\text{ bpw}$ efetivos com grupo 64) sofrem colapso severo ($Q \le 0,1703$), o `Q2_K` do `llama.cpp` retém $Q_{\text{GGUF}} \in [0,4967; 0,7706]$. Isso ocorre porque os esquemas `_K` não são quantizações puras de 2 ou 3 bits: `Q2_K` utiliza **$3,26\text{ a }3,70\text{ bits por peso}$** (mantendo tensores sensíveis de atenção e saída em precisão superior), e `Q3_K_M` utiliza **$4,09\text{ a }4,41\text{ bpw}$** (equivalente em tamanho de arquivo ao 4-bit uniforme do MLX, que usa $4,5\text{ bpw}$).
2. **Vantagem de vazão do MLX em 4 bits no Apple M5 Pro:** comparando configurações de 4 bits nominais (`RTN/AWQ 4b` no MLX vs. `Q4_K_M` no `llama.cpp`), o MLX alcança vazão de decodificação superior em todos os cinco modelos (`400,6` vs. `274,2 tok/s` no `0.6B`; `222,6` vs. `184,1 tok/s` no `1.7B`; `131,5` vs. `116,9 tok/s` no `Llama-3.2-3B`; `104,4` vs. `93,8 tok/s` no `Qwen3-4B`), combinando menor tamanho por peso ($4,5\text{ bpw}$ vs. $\approx 5,0\text{ bpw}$) e menor overhead de desempacotamento do formato afim uniforme nos kernels Metal do MLX.
3. **Ausência de ganho de velocidade do `Q3_K_M` sobre `Q4_K_M`:** em três dos cinco modelos (`Qwen3-0.6B`, `Qwen3-1.7B` e `Qwen3-4B`), o `Q3_K_M` é ligeiramente mais lento na decodificação que o `Q4_K_M` (ex.: `89,4` vs. `93,8 tok/s` no `Qwen3-4B`), corroborando a tese de Benazir e Lin [2026] de que a aritmética extra de reconstrução de máscaras de bits não alinhadas a potências de 2 pode anular a economia marginal de banda de memória.

### 4.6 Concordância $Q \times Q_{\text{acc}}$, Variância de Semente ($\sigma_{\text{cal}}$) e Síntese Geral

A **Tabela 5** consolida os vereditos formais do estudo principal (`protocolo-v2`) e os três diagnósticos quantitativos da Etapa 5.

| Hipótese / Análise | Critério Pré-Registrado | Resultado Empírico | Veredito (v2) |
|---|---|---|---|
| **H1** (Retenção em 4 bits $\ge 0,95$ só com calibração) | $\ge 4/5$ modelos com $Q_{\text{ciente}} \ge 0,95$ e $Q_{\text{RTN}} < 0,95$ | Atendido em **2/4** avaliáveis (`Qwen3-1.7B` AWQ $0,970$; `Llama-3.2-3B` AWQ $0,955$). Em `0.6B` ($0,916$) e `4B` ($0,942$), nenhum método atingiu $0,95$ | **Inconclusiva** |
| **H2** (Sensibilidade por porte na família Qwen3) | Ordem $Q_{0,6\text{B}} < Q_{1,7\text{B}} < Q_{4\text{B}}$ em $\ge 75\%$ das células | Confirmada em **1/2** células do RTN (3 bits: $0,337 < 0,408 < 0,652$; em 4 bits: $0,862 \approx 0,856 < 0,942$) | **Inconclusiva** |
| **H3 — Primária** (Queda superlinear em $L = 1 - Q$ e redução $\ge 20\%$ a 3 bits) | $\Delta L_{3\to 2} > \Delta L_{4\to 3}$ e $L_{3,\text{ciente}} \le 0,8 L_{3,\text{RTN}}$ em $\ge 4$ modelos | Sem SmolLM3: $\Delta L_{4\to 3}=0,389 < \Delta L_{3\to 2}=0,508$, mas só 2 modelos avaliáveis em 3b (`0.6B` e `1.7B`, ambos com redução $\ge 20\%$). Com SmolLM3: efeito teto em $L$ inverte a média ($0,463 > 0,425$) | **Inconclusiva** (Refutada na sensibilidade com SmolLM3) |
| **H3 — Secundária Pós-Hoc** (Escala logarítmica $S = (D_2-D_3)-(D_3-D_4)$ em nats/token) | $\text{IC}_{95\%}(S) > 0$ (sem saturação em $Q \to 0$) | Superlinear nos **5/5** modelos no RTN ($S$ médio $= 9.356$ nats/tok, IC95% $[9.117; 9.596]$) e em todos os métodos calibrados | **Corrobora a 1ª metade** (análise secundária declarada) |
| **H4** (Eficiência de velocidade de decodificação em 4 bits) | $E = G_{\text{dec}}/R_{\text{tam}} \ge 0,60$ nos modelos de 3–4B | `SmolLM3-3B`: $E=0,83$; `Llama-3.2-3B`: $E=0,85$; `Qwen3-4B`: $E=0,87$ (ganho de $3,0\text{–}3,1\times$ sobre BF16) | **Corroborada** |
| **Diagnóstico A** (Concordância $Q \times Q_{\text{acc}}$ zero-shot) | Correlação e coincidência do melhor método em 4 bits | $N=50$ pares; Pearson $r = 0.9520$; Spearman $\rho = 0.9413$; melhor método coincide em **3/4** modelos | **Alta concordância** de ranking |
| **Diagnóstico B** (Variância entre sementes de calibração AWQ 4b) | Sementes $0, 1, 2$ em `Qwen3-0.6B` e `Qwen3-1.7B` | $\sigma_{\text{cal}}(Q) = 0.00649$ ($\text{CV}_{\text{PPL}} = 0,75\%$ no `0.6B` e $0,65\%$ no `1.7B`) | **Baixo ruído de semente** ($< 0,8\%$ em PPL) |
| **Diagnóstico C** (Falha de `NaN` no GPTQ padrão do `mlx-lm 0.32.2`) | Inspeção de autovalores da Hessiana $H = X^\top X$ em `bf16` vs `fp32` | Em `bf16`, `o_proj`/`down_proj` tornam-se indefinidas ($\lambda_{\min} < -\text{damp}$) em 100% dos modelos de 3–4B (9/9 jobs `NaN`); em `fp32`, 0 matrizes falham | **Causa-raiz demonstrada e corrigida (`gptq_fp32`)** |

Os diagnósticos da Etapa 5 fornecem duas garantias metodológicas adicionais:
1. **Concordância entre Perplexidade ($Q$) e Acurácia Zero-Shot ($Q_{\text{acc}}$):** nos $N = 50$ pares válidos da grade v2 (semente 0), a correlação linear de Pearson entre $Q$ e $Q_{\text{acc}}$ é de **$r = 0,9520$** e a correlação de postos de Spearman é de **$\rho = 0,9413$**. Em 4 bits, a escolha do melhor método calibrado por $Q$ coincide com a escolha por $Q_{\text{acc}}$ em **3 dos 4 modelos** (`Qwen3-1.7B`, `Llama-3.2-3B` e `Qwen3-4B`). Observa-se, contudo, que $Q_{\text{acc}}$ é sistematicamente mais indulgente que $Q$: como a acurácia em tarefas de múltipla escolha possui piso no nível do acaso ($\approx 0,25\text{–}0,50$), $Q_{\text{acc}}$ supera $0,95$ em 12 das 13 configurações de 4 bits (inclusive no RTN em quatro modelos), o que demonstra que **a razão de perplexidade $Q$ constitui um teste Popperiano significativamente mais severo e discriminante** para detectar degradação fina na distribuição preditiva.
2. **Estabilidade frente à semente de calibração ($\sigma_{\text{cal}}$):** nas réplicas de semente ($s \in \{0, 1, 2\}$) do AWQ 4 bits, o coeficiente de variação da perplexidade foi de apenas **$0,75\%$** no `Qwen3-0.6B` ($\text{PPL} \in \{23,72; 23,62; 23,96\}$, $\sigma_{\text{cal}}(Q) = 0,00662$) e **$0,65\%$** no `Qwen3-1.7B` ($\text{PPL} \in \{17,23; 17,12; 17,34\}$, $\sigma_{\text{cal}}(Q) = 0,00635$), resultando em um desvio-padrão combinado de calibração $\sigma_{\text{cal}} = 0,00649$. Essa magnitude é incorporada diretamente no alargamento conservador dos intervalos de confiança na extensão `protocolo-v3`:

$$
h_{\text{total}} = \sqrt{h_{\text{boot}}^2 + (1,96 \cdot \sigma_{\text{cal}})^2}
$$

### 4.7 Ameaças à Validade e Limitações

Em aderência ao princípio de transparência científica, registramos cinco limitações do estudo:
1. **Dispositivo único de hardware:** as medições de velocidade, *footprint* e eficiência $E$ referem-se especificamente ao Apple M5 Pro com 24 GiB de memória unificada e não devem ser extrapoladas sem nova medição para GPUs discretas NVIDIA, NPUs móveis ou microcontroladores.
2. **Restrição de memória física para o DWQ em modelos $\ge 3\text{B}$:** como o DWQ requer manter simultaneamente os logits do professor e o grafo de gradientes do aluno para um vocabulário de até 152 mil tokens (atingindo $20,05\text{ GB}$ de *footprint* já no modelo de 0,6B mesmo com alvos pré-calculados em sub-lotes via `dwq_sublote.py`), o método permaneceu restrito aos modelos de 0,6B e 1,7B no hardware de 24 GiB.
3. **Variabilidade térmica nos K-quants do modelo menor (`Qwen3-0.6B`):** embora as 59 medidas de velocidade do MLX e os K-quants dos modelos de 3–4B tenham apresentado variação $\le 1\%$ entre as Rodadas 1 e 2, o `llama-bench` no `Qwen3-0.6B` apresentou variação de $17\text{–}24\%$ entre rodadas, indicando maior sensibilidade de modelos sub-1B a estados dinâmicos de frequência da GPU no `llama.cpp`.
4. **Anomalia isolada de tempo na avaliação de qualidade do `Qwen3-0.6B-fp`:** na primeira execução da fila de qualidade (23/10), o job `Qwen3-0.6B-fp` levou $31\,962\text{ s}$ ($\approx 8\text{ h }53\text{ min}$) por provável contenção de *swap* ou suspensão parcial; como a fase de qualidade é determinística e independente de tempo, e a velocidade oficial foi medida separadamente nas Rodadas 1 e 2 da Fase de Velocidade, nenhum resultado de H1–H4 foi afetado.
5. **Escopo do fichamento bibliográfico:** as 27 fichas em `literatura/fichas/` consolidam metadados verificados via Crossref/arXiv API, resumos oficiais e leitura direta da documentação técnica de repositórios (#26 e #27); a conferência página a página das edições impressas de Popper [1959] e Severino [2017] permanece indicada para citações diretas.

---

## 5. Conclusão e Trabalhos Futuros

Este artigo apresentou um estudo experimental comparativo e pré-registrado de quantização pós-treino (RTN, GPTQ, AWQ, DWQ e K-quants) em pequenos modelos de linguagem ($0,6\text{B–}4\text{B}$) executados localmente em um MacBook Pro Apple M5 Pro com 24 GiB de memória unificada.

Retomando o problema de pesquisa e as quatro hipóteses falseáveis formuladas sob o referencial de Popper e Severino, concluímos que:
1. **A conversão de compressão em velocidade de decodificação na memória unificada (H4) é corroborada com folga** ($E = 0,83\text{ a }0,87 \ge 0,60$ nos modelos de 3–4B), entregando aceleração de $2,93\times\text{ a }3,09\times$ em 4 bits no MLX com redução de $3,5\times$ no tamanho em disco e nos picos de memória de GPU.
2. **A implementação padrão do GPTQ no `mlx-lm 0.32.2` contém uma falha numérica crítica em modelos de 3–4B:** o acúmulo de $X^\top X$ em `bfloat16` torna a matriz Hessiana indefinida nas projeções `o_proj` e `down_proj`, levando a fatoração de Cholesky a produzir `NaN` em 100% dos testes de 8, 4 e 2 bits nos modelos de 3–4B. A acumulação em `float32` (`gptq_fp32`), aliada à execução camada a camada (`gptq_camada.py`), corrige o problema na origem e reduz o consumo de memória em $3\text{–}4\times$.
3. **Em 3 bits, algoritmos cientes de dados proporcionam ganhos decisivos sobre o RTN:** no MLX, o AWQ eleva a retenção de qualidade $Q$ de $0,6394 \to 0,8090$ no `Llama-3.2-3B` e de $0,6523 \to 0,8069$ no `Qwen3-4B`, enquanto o DWQ reduz a perda em $31\%\text{–}40\%$ nos modelos de $0,6\text{B–}1,7\text{B}$. No `llama.cpp`, a alocação mista de precisão dos K-quants (`Q4_K_M`, `Q3_K_M`, `Q2_K` com `imatrix`) evita o colapso catastrófico em baixas larguras nominais às custas de um maior número médio de bits por peso e menor vazão de decodificação que o MLX em 4 bits.
4. **O pré-registro Popperiano cumpriu seu papel demarcador:** impediu que o efeito teto da métrica linear $L = 1 - Q$ aos 2 bits ou a falha de `NaN` do GPTQ fossem mascarados por alterações retrospectivas de regras, separando de forma transparente os vereditos originais da Matriz v2 (H1–H3 inconclusivas, H4 corroborada), a análise pós-hoc em escala de NLL (que confirma a superlinearidade em 5/5 modelos) e a extensão pré-registrada `protocolo-v3`.

Como **trabalhos futuros**, destacam-se: (i) a conclusão da fila do Bloco 1 da extensão `protocolo-v3` (`gptq_fp32` em 4 e 3 bits nos modelos de 1,7B a 4B via `experimentos/rodar_bloco1_v3.sh`); (ii) a submissão de um *pull request* upstream ao repositório `ml-explore/mlx-lm` propondo a acumulação em `float32` no `Catcher` do GPTQ e a coleta explícita de `input_feat` por bloco no AWQ; e (iii) a avaliação conjunta de quantização de pesos e quantização `int4` do *KV-cache* em contextos longos ($\ge 32\text{k}$ tokens) em memória unificada.

---

## Referências Bibliográficas (ABNT NBR 6023)

- BARRIOS, Wayner. **Native LLM and MLLM Inference at Scale on Apple Silicon**. Preprint arXiv:2601.19139, 2026. *(Preprint sem revisão por pares localizada)*.
- BENAZIR, Afsara; LIN, Felix Xiaozhu. Benchmarking and Characterization of Large Language Model Inference on Apple Silicon. **ACM SIGMETRICS Performance Evaluation Review**, 2026. DOI: `10.1145/3820171.3806865`. *(Preprint arXiv:2508.08531)*.
- BERGACH, Mohamed Amine. **When Quantization Is Free: An int4 KV Cache That Outruns fp16 on Apple Silicon**. Preprint arXiv:2605.05699, 2026. *(Preprint de autor único)*.
- BORGERSEN, Karl Audun; GOODWIN, Morten. English K\_Quantization of LLMs Does Not Disproportionately Diminish Multilingual Performance. In: **Artificial Intelligence XLII** (Lecture Notes in Computer Science). Cham: Springer Nature Switzerland, 2025. DOI: `10.1007/978-3-032-11402-0_8`.
- CHAI, Yuji; KWUN, Mujin; BROOKS, David; WEI, Gu-Yeon. **FlexQuant: Elastic Quantization Framework for Locally Hosted LLM on Edge Devices**. Preprint arXiv:2501.07139, 2025.
- DANTAS, Pierre Vilar; SABINO DA SILVA, Waldir; CORDEIRO, Lucas Carvalho et al. A comprehensive review of model compression techniques in machine learning. **Applied Intelligence**, v. 54, n. 22, 2024. DOI: `10.1007/s10489-024-05747-w`.
- DETTMERS, Tim; PAGNONI, Artidoro; HOLTZMAN, Ari; ZETTLEMOYER, Luke. QLoRA: Efficient Finetuning of Quantized LLMs. **arXiv preprint arXiv:2305.14314**, 2023. *(Conferência NeurIPS 2023 a conferir na versão final)*.
- EGIAZARIAN, Vage et al. **Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization**. Preprint arXiv:2509.23202, 2025.
- FRANTAR, Elias; ASHKBOOS, Saleh; HOEFLER, Torsten; ALISTARH, Dan. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. **arXiv preprint arXiv:2210.17323**, 2023. *(Conferência ICLR 2023 a conferir na versão final)*.
- GHOLAMI, Amir; KIM, Sehoon; DONG, Zhen; YAO, Zhewei; MAHONEY, Michael W.; KEUTZER, Kurt. A Survey of Quantization Methods for Efficient Neural Network Inference. In: **Low-Power Computer Vision**. Chapman and Hall/CRC, 2022. *(Preprint arXiv:2103.13630)*.
- HU, Jundong; RAMACHANDRAN, Shekar. **The Structure of Quantization Damage in LLMs: Why the Next Bit Should Be Spent Globally**. Preprint arXiv:2609.01587, 2026.
- JAVAT, Abdurrahman; KAZAKOV, Allan. **Silicon Showdown: Performance, Efficiency, and Ecosystem Barriers in Consumer-Grade LLM Inference**. Preprint arXiv:2605.00519, 2026.
- KAWRAKOW, Iwan. **k-quants: 2–6 bit quantization methods and mixes (Pull Request #1684, llama.cpp)**. GitHub: `ggml-org/llama.cpp`, 2023. Disponível em: `https://github.com/ggml-org/llama.cpp/pull/1684`. Acesso em: 22 out. 2026.
- LIN, Ji; CHEN, Wei-Ming; CAI, Han; GAN, Chuang; HAN, Song. MCUNetV2: Memory-Efficient Patch-based Inference for Tiny Deep Learning. **arXiv preprint arXiv:2110.15352**, 2021. *(Conferência NeurIPS 2021 a conferir)*.
- LIN, Ji; ZHU, Ligeng; CHEN, Wei-Ming; WANG, Wei-Chen; HAN, Song. Tiny Machine Learning: Progress and Futures. **IEEE Circuits and Systems Magazine**, v. 23, n. 3, p. 8–34, 2023. DOI: `10.1109/MCAS.2023.3302182`.
- LIN, Ji et al. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. **arXiv preprint arXiv:2306.00978**, 2024. *(Conferência MLSys 2024 a conferir)*.
- MA, Shuming; WANG, Hongyu; MA, Lingxiao; WANG, Lei et al. **The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits**. Preprint arXiv:2402.17764, 2024.
- MENGHANI, Gaurav. Efficient Deep Learning: A Survey on Making Deep Learning Models Smaller, Faster, and Better. **ACM Computing Surveys**, v. 55, n. 12, p. 1–37, 2023. DOI: `10.1145/3578938`.
- MLX TEAM. **Learned Quantization (DWQ, AWQ, dynamic quantization, GPTQ) — Documentação do mlx-lm**. GitHub: `ml-explore/mlx-lm`, 2026. Disponível em: `https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md`. Acesso em: 22 out. 2026.
- NAGEL, Markus; FOURNARAKIS, Marios; AMJAD, Rana Ali; BONDARENKO, Yelysei; VAN BAALEN, Mart; BLANKEVOORT, Tijmen. **A White Paper on Neural Network Quantization**. Preprint arXiv:2106.08295, 2021.
- POPPER, Karl Raimund. **A lógica da pesquisa científica** (*The Logic of Scientific Discovery*, 1959; orig. *Logik der Forschung*, 1934). São Paulo: Cultrix. *(Conferir edição e paginação `[pág.]` antes da submissão)*.
- RAJESH, Varun et al. **Production-Grade Local LLM Inference on Apple Silicon: A Comparative Study of MLX, MLC-LLM, Ollama, llama.cpp, and PyTorch MPS**. Preprint arXiv:2511.05502, 2025.
- SEVERINO, Antônio Joaquim. **Metodologia do trabalho científico**. São Paulo: Cortez Editora. *(Conferir edição e paginação `[pág.]` antes da submissão)*.
- SHUVO, Md Maruf Hossain; ISLAM, Syed Kamrul; CHENG, Jianlin; MORSHED, Bashir I. Efficient Acceleration of Deep Learning Inference on Resource-Constrained Edge Devices: A Review. **Proceedings of the IEEE**, v. 111, n. 1, p. 42–91, 2023. DOI: `10.1109/JPROC.2022.3226481`.
- TAHERIN, Amir et al. **Hydra: Phase-Aware Workload Characterization of LLM Inference across Edge SoC Generations, Backends, and Quantization Levels**. Preprint arXiv:2608.25053, 2026.
- TU, Fangbo et al. **Beyond Output Matching: Preserving Internal Geometry in NVFP4 LLM Distillation**. Preprint arXiv:2606.05682, 2026.
- XIAO, Guangxuan; LIN, Ji; SEZNEC, Mickael; WU, Hao; DEMOUTH, Julien; HAN, Song. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. **arXiv preprint arXiv:2211.10438**, 2023. *(Conferência ICML 2023 a conferir)*.
- XU, Weili et al. **QATFactory: A Versatile, Deployment-Aligned Framework for Quantization-aware Training and Distillation of LLMs**. Preprint arXiv:2609.39223, 2026.
- ZHU, Xunyu; LI, Jian; LIU, Yong; MA, Can; WANG, Weiping. A Survey on Model Compression for Large Language Models. **Transactions of the Association for Computational Linguistics (TACL)**, v. 12, p. 1556–1577, 2024. DOI: `10.1162/tacl_a_00704`.
