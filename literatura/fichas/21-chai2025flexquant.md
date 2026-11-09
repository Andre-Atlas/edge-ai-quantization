# Ficha de leitura — #21 Chai et al. (2025)

**Referência (ABNT):** CHAI, Yuji; KWUN, Mujin; BROOKS, David; WEI, Gu-Yeon. FlexQuant: Elastic Quantization Framework for Locally Hosted LLM on Edge Devices. **arXiv preprint arXiv:2501.07139**, 2025. Disponível em: https://arxiv.org/abs/2501.07139. Acesso em: 22 out. 2026.
**Chave BibTeX:** `chai2025flexquant`
**Data da leitura:** 2026-11-09
**Escopo da leitura:** Consolidação dos metadados e resumo verificados via API do arXiv (`2501.07139` v2; nota de desambiguação: existe outro artigo homônimo "FlexQuant" no EMNLP Findings 2025 com DOI `10.18653/v1/2025.findings-emnlp.221` e título/autores diferentes, que não deve ser confundido com este trabalho de Harvard), sem fabricação de números de página ou citações literais entre aspas de PDF não lido na íntegra.

## 1. Análise textual (visão global)
- **Tipo de texto:** Preprint de arquitetura de sistemas e compressão elástica (grupo de arquitetura de computadores de Harvard — David Brooks e Gu-Yeon Wei).
- **Estrutura conceitual:** Caracterização do problema da memória unificada compartilhada em dispositivos de borda (onde a memória livre flutua dinamicamente conforme outros aplicativos do usuário abrem e fecham), seguida da proposta do framework `FlexQuant`, que alterna elasticamente entre configurações quantizadas de diferentes tamanhos conforme a disponibilidade instantânea de RAM.
- **Vocabulário-chave:** *elastic quantization*, *unified memory*, *fluctuating available memory*, *locally hosted LLM*, *edge devices*, *Pareto frontier*, *dynamic model switching*.

## 2. Análise temática
- **Tema:** Quantização elástica de LLMs hospedados localmente em dispositivos de borda com memória unificada e disponibilidade dinâmica de RAM.
- **Problema:** Os métodos tradicionais de quantização pós-treino assumem um orçamento de memória estático; contudo, em dispositivos reais do usuário (como laptops e smartphones com memória unificada compartilhada entre CPU, GPU e sistema operacional), a memória disponível para o LLM varia ao longo do tempo devido às demais aplicações em execução.
- **Tese / hipótese do(s) autor(es):** Em vez de implantar um único modelo quantizado estático (que pode sofrer paginação/falha quando a memória livre cai ou desperdiçar qualidade quando há memória ociosa), um framework elástico que opera sobre a fronteira de Pareto qualidade × memória permite adaptar o nível de quantização dinamicamente às flutuações da memória compartilhada.
- **Argumentação / método:** Os autores modelam a flutuação de memória disponível em dispositivos de borda e constroem um conjunto elástico de variantes quantizadas selecionadas sobre a fronteira de Pareto para maximizar a qualidade instantânea sem violar o teto de memória livre.
- **Resultados principais:** O `FlexQuant` demonstra que a adaptação elástica do bit-width ao orçamento instantâneo de memória evita quedas abruptas de desempenho do sistema operacional (swapping/OOM) enquanto preserva qualidade superior ao modelo estático mais conservador.

## 3. Análise interpretativa
- **Contexto e relevância:** Introduz na literatura um conceito essencial para computadores pessoais com memória unificada (como o MacBook Pro M5 Pro): na prática real, um dispositivo de 24 GiB de memória unificada **não** dispõe de 24 GiB estáticos dedicados ao LLM, pois o macOS, o compositor gráfico e os aplicativos concorrentes disputam o mesmo pool físico de DRAM.
- **Pressupostos:** Pressupõe que se conhece previamente a fronteira de Pareto empírica entre tamanho de memória e qualidade para os diferentes níveis de bits e algoritmos de quantização no dispositivo alvo.
- **Pontos fortes:** Conecta diretamente a construção da fronteira de Pareto qualidade × tamanho (que calculamos formalmente em `experimentos/analise.py`) com a tomada de decisão operacional em memória unificada.
- **Limites:** Preprint sem veículo confirmado no Crossref (exigindo cuidado na citação para não confundir com o homônimo do EMNLP Findings 2025); foca no mecanismo elástico de seleção/troca e não na comparação exaustiva entre `RTN`, `GPTQ`, `AWQ`, `DWQ` e `K-quants`.

## 4. Problematização
Questões e conexões diretas que o texto suscita para o nosso estudo experimental no Apple M5 Pro (24 GiB):
- **Justificativa para o guarda de memória (`18 GiB`) e para a fronteira de Pareto no M5 Pro:** Em nosso protocolo experimental (`experimentos/rodar.py` e `protocolo-v2.md`), estabelecemos um teto operacional de segurança de $18\text{ GiB}$ de memória ativa dentro dos $24\text{ GiB}$ físicos do MacBook Pro M5 Pro justamente porque a memória unificada é compartilhada com o sistema operacional. Chai et al. (2025) dão suporte teórico direto a essa restrição de projeto.
- **Aplicação direta da nossa fronteira de Pareto (`hipoteses.json` → `"pareto"`):** Nosso script `analise.py` extrai explicitamente a fronteira de Pareto $(M, Q)$ para cada um dos 5 modelos entre 2, 3, 4, 8 e 16 bits. Os dados que obtivemos no M5 Pro mostram exatamente quais configurações deveriam compor um catálogo elástico à la `FlexQuant`:
  - No `Qwen3-1.7B`, por exemplo, a fronteira de Pareto em `hipoteses.json` é composta por `DWQ 2b` ($0{,}55\text{ GB}$, $Q=0{,}0037$), `DWQ 3b` ($0{,}77\text{ GB}$, $Q=0{,}6438$), `AWQ 3b` ($0{,}82\text{ GB}$, $Q=0{,}7903$), `DWQ 4b` ($0{,}98\text{ GB}$, $Q=0{,}9336$), `AWQ 4b` ($1{,}00\text{ GB}$, $Q=0{,}9767$) e `AWQ 8b` ($1{,}70\text{ GB}$, $Q=1{,}0007$).
  - Além disso, nossa comparação entre tamanhos de modelo mostra que, sob restrição severa de memória ($\approx 1{,}8\text{ GB}$), um modelo maior em 4 bits (`Llama-3.2-3B AWQ 4b`, $1{,}85\text{ GB}$, $\text{PPL} = 8{,}20$) supera com folga um modelo menor em 8 bits (`Qwen3-1.7B 8b`, $1{,}70\text{ GB}$, $\text{PPL} = 16{,}57$).

## 5. Síntese pessoal
Chai et al. (2025) transformam a memória unificada de uma simples constante de capacidade em uma restrição dinâmica e flutuante. Em nosso artigo, esse trabalho valoriza diretamente a fronteira de Pareto que construímos no M5 Pro: demonstramos empiricamente quais algoritmos (`AWQ` em 3–4 bits, `GPTQ-fp32` em 4 bits e `K-quants` `Q4_K_M`/`Q3_K_M`) dominam a fronteira de Pareto e devem ser escolhidos conforme a pressão instantânea de memória no dispositivo.

## 6. Falseabilidade (Popper)
- **A hipótese central do trabalho é falseável?** Sim. A proposição de que a seleção elástica de modelos quantizados sobre a fronteira de Pareto supera modelos estáticos sob flutuação de memória em dispositivos de borda pode ser refutada se o sobrecusto de transição/armazenamento anular o ganho de qualidade ou violar os limites de latência.
- **Como foi testada?** Simulando e executando cargas concorrentes de memória em dispositivos de borda com LLMs localmente hospedados e medindo qualidade média e eventos de contenção de memória.
- **Quais condições limitam a conclusão?** Exige armazenamento secundário (SSD) rápido ou compartilhamento de pesos em memória para que a troca entre níveis de bits não introduza latência proibitiva.

## Citações úteis / Pontos-chave verificados para citação indireta
- **Seções de uso previstas no artigo:** Discussão (`D`) e Trabalhos Futuros (`F`).
- **Ponto verificado 1 (Memória unificada como recurso dinâmico e compartilhado):** Em dispositivos de borda onde CPU, GPU e sistema operacional compartilham a mesma memória física, a memória livre para o LLM flutua continuamente, exigindo mapear a fronteira de Pareto entre pegada de memória e qualidade nos diferentes níveis de quantização (CHAI et al., 2025).
- **Ponto verificado 2 (Insumo empírico para quantização elástica):** Nossa caracterização empírica da fronteira de Pareto no Apple M5 Pro (de 2 a 16 bits nos 5 modelos) fornece os pontos ótimos de operação necessários para políticas de adaptação elástica na borda (CHAI et al., 2025).
