# Ficha de leitura — #25 Taherin et al. (2026)

**Referência (ABNT):** TAHERIN, Amir et al. Hydra: Phase-Aware Workload Characterization of LLM Inference across Edge SoC Generations, Backends, and Quantization Levels. **arXiv preprint arXiv:2608.25053**, 2026. Disponível em: https://arxiv.org/abs/2608.25053. Acesso em: 22 out. 2026. (Nota: aceito no IEEE International Symposium on Workload Characterization — IISWC 2026 segundo comentário do arXiv; paginação e DOI a conferir).
**Chave BibTeX:** `taherin2026hydra`
**Data da leitura:** 2026-11-09
**Escopo da leitura:** Consolidação dos metadados e resumo verificados via API do arXiv (`2608.25053`, com indicação de aceite no IISWC 2026 nos metadados do autor no arXiv), sem fabricação de números de página ou citações literais entre aspas de PDF não lido na íntegra.

## 1. Análise textual (visão global)
- **Tipo de texto:** Artigo de caracterização de cargas de trabalho (*workload characterization*) e *benchmarking* de borda (grupo de David Kaeli e Yanzhi Wang, Northeastern University).
- **Estrutura conceitual:** Proposição da suíte e metodologia `Hydra` para avaliação desagregada por fase (*phase-aware*: fase de processamento de prompt/*prefill* separada da fase de geração autorregressiva/*decode*) cruzando três dimensões: (i) gerações de SoCs de borda (NVIDIA Jetson Xavier, Orin e Thor), (ii) backends de execução (incluindo `llama.cpp`), e (iii) níveis de quantização em 13 LLMs.
- **Vocabulário-chave:** *phase-aware characterization*, *prefill phase*, *decode phase*, *edge SoC*, *quantization levels*, *llama.cpp*, *compute-bound vs. memory-bound*.

## 2. Análise temática
- **Tema:** Caracterização sistemática e sensível à fase (*prefill* vs. *decode*) da inferência de LLMs através de gerações de SoCs de borda, backends de software e níveis de quantização.
- **Problema:** Avaliações de inferência na borda frequentemente reportam apenas latência ponta a ponta (*end-to-end latency*) ou vazão agregada, misturando duas fases com naturezas físicas opostas — o *prefill* paralelo (limitado por cômputo matricial) e o *decode* sequencial (limitado por largura de banda de memória) — o que obscurece o real impacto de cada nível de quantização sobre o hardware.
- **Tese / hipótese do(s) autor(es):** A caracterização rigorosa de inferência de LLMs em SoCs de borda exige separar estruturalmente as métricas de *prefill* e *decode* ao cruzar níveis de quantização e backends, pois a quantização interage de forma assimétrica com cada fase e geração de SoC.
- **Argumentação / método:** Os autores constroem o benchmark `Hydra` e avaliam 13 LLMs ao longo de 3 gerações de SoCs NVIDIA Jetson (Xavier, Orin e Thor) e 5 formatos/backends de execução (incluindo `llama.cpp` com diferentes quantizações), instrumentando separadamente as fases de *prefill* e *decode*.
- **Resultados principais:** O estudo comprova que a separação por fase é indispensável: enquanto a fase de *decode* escala fortemente com a redução do tráfego de memória proporcionada pela quantização, a fase de *prefill* exibe sensibilidade distinta dependente dos kernels de desempacotamento e das unidades matriciais de cada SoC e backend.

## 3. Análise interpretativa
- **Contexto e relevância:** Representa o estado da arte de 2026 em metodologia de *benchmarking* de inferência quantizada em SoCs de borda. Embora o hardware avaliado pelos autores seja a linha NVIDIA Jetson (Xavier/Orin/Thor) e não Apple Silicon, a **metodologia experimental sensível à fase (*phase-aware*)** com artefato reproduzível é exatamente o padrão metodológico adotado em nosso protocolo no M5 Pro.
- **Pressupostos:** Assume que SoCs de borda compartilham restrições de memória e potência que tornam a distinção *prefill*/*decode* crítica para o projeto de sistemas.
- **Pontos fortes:** Valida na literatura de arquitetura de computadores (IISWC 2026) a decisão metodológica de medir e reportar `prefill_tok_s` e `decode_tok_s` separadamente, bem como testar a Hipótese H4 especificamente sobre a fase de *decode*.
- **Limites:** Avalia apenas SoCs NVIDIA Jetson (baseados em arquitetura ARM + GPU CUDA integrada), sem incluir Apple Silicon nem comparar a qualidade de `GPTQ`, `AWQ` e `DWQ` no `MLX`.

## 4. Problematização
Questões e conexões diretas que o texto suscita para o nosso estudo experimental no Apple M5 Pro (24 GiB):
- **Respaldo direto para o protocolo de medição da nossa Hipótese H4 (`prefill` vs. `decode`):** Em nosso código experimental (`experimentos/rodar.py` e `experimentos/kquant.py`), separamos estritamente a medição de `prefill_tok_s` (512 tokens de entrada) e `decode_tok_s` (256 tokens gerados, mediana de 5 repetições após aquecimento) e formulamos a H4 especificamente sobre a vazão de **decodificação** (`decode_tok_s`). Taherin et al. (2026) fundamentam metodologicamente por que não se deve misturar *prefill* e *decode* ao avaliar o ganho de velocidade da quantização.
- **Confirmação empírica da assimetria *prefill*/*decode* no Apple M5 Pro:** Nossos dados no M5 Pro reproduzem em Apple Silicon o fenômeno caracterizado pelo `Hydra` em SoCs Jetson:
  - Na fase de **decode** (limitada por largura de banda), a quantização de 16 bits para 4 bits acelera o `Llama-3.2-3B` de $\approx 23{,}8\text{ tok/s}$ para $\approx 68{,}8\text{ tok/s}$ ($2{,}89\times$) e o `Qwen3-4B` de $\approx 18{,}5\text{ tok/s}$ para $\approx 56{,}5\text{ tok/s}$ ($3{,}05\times$), corroborando a H4 ($E = 0{,}85\text{–}0{,}87$).
  - Já na fase de **prefill**, observamos que em vários casos o modelo quantizado em 2 ou 3 bits (especialmente nos `K-quants` complexos `Q2_K` e `Q3_K_M` do `llama.cpp`) tem *prefill* mais lento ou próximo ao `F16` devido ao sobrecusto de desempacotamento por bloco nas unidades de cômputo.

## 5. Síntese pessoal
Taherin et al. (2026) consolidam no benchmark `Hydra` a diretriz metodológica de que toda avaliação rigorosa de LLMs quantizados na borda deve ser *phase-aware*, separando o *prefill* (limitado por cômputo e sensível ao custo de dequantização) do *decode* (limitado por banda de memória). Em nosso artigo, este trabalho respalda diretamente o desenho do nosso protocolo experimental para a H4 e contextualiza nossos resultados no M5 Pro frente a outros SoCs de borda modernos (Jetson Orin/Thor).

## 6. Falseabilidade (Popper)
- **A hipótese central do trabalho é falseável?** Sim. A proposição de que os níveis de quantização afetam de modo qualitativamente distinto as fases de *prefill* e *decode* através de gerações de SoCs e backends seria refutada se ambas as fases escalassem na mesma proporção com a redução de bits.
- **Como foi testada?** Medindo separadamente a latência/vazão de *prefill* e *decode* de 13 LLMs em 3 gerações de SoCs Jetson e 5 configurações de execução/quantização.
- **Quais condições limitam a conclusão?** Os valores absolutos medidos no `Hydra` referem-se à arquitetura NVIDIA Jetson; a magnitude exata da razão *prefill*/*decode* em Apple Silicon depende dos kernels Metal de `mlx-lm` e `llama.cpp`, que nosso estudo mede diretamente no M5 Pro.

## Citações úteis / Pontos-chave verificados para citação indireta
- **Seções de uso previstas no artigo:** Metodologia (`M`) e Discussão (`D`).
- **Ponto verificado 1 (Protocolo de medição *phase-aware*):** Avaliações rigorosas de inferência quantizada em SoCs de borda devem desagregar as fases de *prefill* (processamento paralelo do prompt, limitado por cômputo) e *decode* (geração autorregressiva, limitada por largura de banda de memória), pois a quantização atua de maneira distinta sobre cada regime (TAHERIN et al., 2026).
- **Ponto verificado 2 (Paralelo Jetson vs. Apple Silicon):** A metodologia *phase-aware* proposta por Taherin et al. (2026) em SoCs NVIDIA Jetson com `llama.cpp` alinha-se ao nosso protocolo no Apple M5 Pro, onde isolamos `prefill_tok_s` (512 tokens) e `decode_tok_s` (256 tokens) tanto em `mlx-lm` quanto em `llama.cpp`.
