# Ficha de leitura — #15 Rajesh et al. (2025)

**Referência (ABNT):** RAJESH, Varun et al. Production-Grade Local LLM Inference on Apple Silicon: A Comparative Study of MLX, MLC-LLM, Ollama, llama.cpp, and PyTorch MPS. **arXiv preprint arXiv:2511.05502**, 2025. Disponível em: https://arxiv.org/abs/2511.05502. Acesso em: 22 out. 2026.
**Chave BibTeX:** `rajesh2025production`
**Data da leitura:** 2026-11-09
**Escopo da leitura:** Consolidação dos metadados e resumo verificados via API do arXiv (`2511.05502`, sem versão publicada localizada no Crossref), sem fabricação de números de página ou citações literais entre aspas de PDF não lido na íntegra.

## 1. Análise textual (visão global)
- **Tipo de texto:** Preprint de avaliação empírica comparativa de motores de inferência (*inference engines* / *runtimes*) locais em Apple Silicon.
- **Estrutura conceitual:** Comparação horizontal de cinco pilhas de software para execução local de LLMs em macOS (`MLX`, `MLC-LLM`, `Ollama`, `llama.cpp` e `PyTorch MPS`), avaliando métricas operacionais de produção (tempo até o primeiro token, vazão de decodificação, escalabilidade em contexto longo e suporte a formatos de quantização).
- **Vocabulário-chave:** *local LLM inference*, *MLX*, *llama.cpp*, *MLC-LLM*, *PyTorch MPS*, *time-to-first-token (TTFT)*, *decode throughput*, *Apple M2 Ultra*.

## 2. Análise temática
- **Tema:** Avaliação comparativa de desempenho e maturidade de cinco frameworks de inferência local de LLMs executados sobre hardware Apple Silicon.
- **Problema:** Desenvolvedores e pesquisadores que implantam LLMs localmente em Apple Silicon deparam-se com múltiplos backends concorrentes (`MLX`, `llama.cpp`, `MLC-LLM`, `Ollama`, `PyTorch MPS`), cada qual com abstrações de grafo, kernels Metal e formatos de quantização distintos, faltando um comparativo padronizado no mesmo hardware.
- **Tese / hipótese do(s) autor(es):** Runtimes nativamente desenhados para Apple Silicon ou com kernels Metal altamente especializados (`MLX`, `llama.cpp`, `MLC-LLM`) superam substancialmente camadas genéricas de compatibilidade como `PyTorch MPS` em vazão, latência inicial (TTFT) e eficiência no suporte a modelos quantizados.
- **Argumentação / método:** Os autores executam experimentos sistemáticos em uma estação Mac Studio equipada com Apple M2 Ultra (192 GB de memória unificada), medindo TTFT, vazão de geração, comportamento em contextos longos e compatibilidade com diferentes níveis de quantização nos cinco frameworks.
- **Resultados principais:** Os resultados destacam a superioridade de `MLX` e `llama.cpp` (e derivados) frente ao backend `PyTorch MPS` para inferência quantizada em Apple Silicon, evidenciando diferenças de especialização entre o ecossistema nativo da Apple (`MLX`) e o ecossistema portátil `GGUF` (`llama.cpp`).

## 3. Análise interpretativa
- **Contexto e relevância:** Representa um marco importante na literatura recente (2025) por documentar empiricamente a transição do ecossistema local em macOS do `PyTorch MPS` para `MLX` e `llama.cpp`.
- **Pressupostos:** O trabalho trata os *runtimes* como unidade primária de análise e utiliza uma estação de trabalho de topo de linha (M2 Ultra com 192 GB), onde restrições severas de memória (como o limite de 24 GiB de um laptop de borda) não são o fator limitante imediato para modelos pequenos.
- **Pontos fortes:** Abrange simultaneamente os dois motores utilizados em nosso projeto (`MLX` e `llama.cpp`), servindo como justificativa metodológica externa para a seleção de ambos como representantes do estado da arte em Apple Silicon.
- **Limites:** Por ser um preprint sem revisão por pares confirmada e focado na comparação de *runtimes* em hardware de 192 GB, não investiga a qualidade estatística (perplexidade e acurácia zero-shot) nem a estabilidade numérica de diferentes algoritmos de PTQ (`RTN`, `GPTQ`, `AWQ`, `DWQ`, `K-quants`) dentro de cada motor.

## 4. Problematização
Questões e conexões diretas que o texto suscita para o nosso estudo experimental no Apple M5 Pro (24 GiB):
- **Justificativa da dupla escolha de motores (`mlx-lm` e `llama.cpp`):** Nosso estudo adota exatamente os dois líderes identificados por Rajesh et al. (2025) para Apple Silicon: `mlx-lm` (para comparar `RTN`, `GPTQ`, `AWQ` e `DWQ` no mesmo formato de empacotamento afim de grupo 64) e `llama.cpp` (para avaliar os `K-quants` `Q8_0`, `Q4_K_M`, `Q3_K_M` e `Q2_K` com *importance matrix*).
- **Contraste de escala de hardware (192 GB vs. 24 GiB) e falha do DWQ em 3–4B:** Enquanto no M2 Ultra (192 GB) de Rajesh et al. (2025) a memória folgada oculta picos de alocação, em nosso MacBook Pro M5 Pro com 24 GiB documentamos uma barreira prática crucial: métodos de destilação de quantização como o `DWQ` exigem manter simultaneamente na memória unificada o modelo professor em `bf16`, o modelo aluno quantizado e os gradientes/estados do otimizador Adam, inviabilizando sua execução nos modelos de 3B e 4B sob o teto seguro de 18 GiB (Seção 1 de `protocolo-v3.md`).
- **Comparação de velocidade `MLX` vs. `llama.cpp` (H4):** Nas nossas medições de velocidade no M5 Pro (`resultados.csv` vs. `kquant-velocidade.csv`), observamos diferenças sistemáticas de *prefill* e *decode* entre `mlx-lm` (pesos afins uniformes por grupo) e `llama.cpp` (superblocos mistos `K-quants`), alinhadas à caracterização de runtimes de Rajesh et al. (2025).

## 5. Síntese pessoal
Rajesh et al. (2025) validam a escolha arquitetural do nosso protocolo experimental ao demonstrar que `MLX` e `llama.cpp` constituem os dois pilares de inferência de alto desempenho em Apple Silicon, superando o `PyTorch MPS`. Ao mesmo tempo, o contraste entre o ambiente de 192 GB dos autores e o nosso laptop M5 Pro de 24 GiB evidencia a contribuição original do nosso trabalho: expor os limites de memória durante a própria etapa de calibração/quantização (`DWQ` e `GPTQ`) e comparar rigorosamente a qualidade dos algoritmos de PTQ.

## 6. Falseabilidade (Popper)
- **A hipótese central do trabalho é falseável?** Sim. A afirmação de que runtimes nativos (`MLX`, `llama.cpp`, `MLC-LLM`) dominam `PyTorch MPS` em TTFT e vazão de decodificação em Apple Silicon é diretamente refutável por medições cronometradas sob o mesmo modelo, prompt e comprimento de geração.
- **Como foi testada?** Mediante benchmarks comparativos dos cinco motores em um Mac Studio M2 Ultra de 192 GB sob diferentes comprimentos de contexto e configurações de quantização.
- **Quais condições limitam a conclusão?** Os resultados refletem as versões específicas de cada framework em 2025 e um chip da geração M2 Ultra (duplo die com altíssima largura de banda); a hierarquia relativa de desempenho entre `MLX` e `llama.cpp` pode variar em chips mais recentes (como o M5 Pro) e conforme o esquema exato de bits (`g64` afim vs. `Q4_K_M`).

## Citações úteis / Pontos-chave verificados para citação indireta
- **Seções de uso previstas no artigo:** Referencial Teórico (`R`), Metodologia (`M`) e Discussão (`D`).
- **Ponto verificado 1 (Seleção de runtimes):** Em avaliações comparativas de inferência local em Apple Silicon, `MLX` e `llama.cpp` destacam-se como os motores de referência para execução quantizada frente a alternativas baseadas em `PyTorch MPS` (RAJESH et al., 2025).
- **Ponto verificado 2 (Runtimes vs. Algoritmos de PTQ):** A literatura recente sobre Apple Silicon concentra-se na comparação de motores de execução em estações de alta capacidade (como M2 Ultra de 192 GB), deixando em aberto a avaliação controlada de algoritmos de quantização pós-treino (`RTN`, `GPTQ`, `AWQ`, `DWQ`, `K-quants`) em dispositivos portáteis de 24 GiB (RAJESH et al., 2025).
