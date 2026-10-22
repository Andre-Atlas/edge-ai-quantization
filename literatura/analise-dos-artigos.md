# Análise dos artigos: pontos relevantes, diferenças e uso no artigo

> Baseada em resumos e conhecimento prévio; **nenhum artigo foi lido na íntegra**. Confirmar nas fichas de leitura (`metodologia/ficha-modelo.md`).

## Surveys (referencial teórico)
| Trabalho | Ponto central | Uso no artigo |
|---|---|---|
| Menghani 2023 (ACM CSUR) | Taxonomia de *Efficient DL*: compressão, aprendizado eficiente, automação, arquiteturas, infraestrutura | Esqueleto da seção de referencial; situa a quantização no todo |
| Shuvo 2023 (Proc. IEEE) | Inferência em dispositivos restritos, co-projeto HW/SW e aceleradores (mais orientado a hardware) | Embasa as métricas de borda e a escolha do dispositivo |
| Dantas 2024 (Springer) | Revisão geral de poda, quantização, destilação | Contexto e justificativa do recorte (só quantização) |
| Gholami 2022 | Survey específico de quantização (uniforme/não uniforme, simétrica, PTQ/QAT, bit misto) | Vocabulário e definições formais |
| Nagel 2021 | Receitas práticas de PTQ/QAT para implantação | Protocolo experimental (calibração, granularidade) |
| Zhu 2024 (TACL) | Compressão específica para LLMs | Ponte surveys gerais → métodos de LLM |
| Lin 2023 (IEEE CAS) | Panorama TinyML e co-projeto algoritmo–sistema | Contexto do extremo da borda; trabalhos futuros |

## Métodos
- **GPTQ × AWQ:** ambos pós-treino e só de pesos. GPTQ reconstrói os pesos camada a camada usando a Hessiana; AWQ não reconstrói e protege canais salientes por escala guiada por ativações, o que o torna, segundo seus autores, menos sujeito a *overfitting* à calibração. *Uso:* núcleo da comparação (H1, H3).
- **SmoothQuant:** quantiza também ativações (W8A8) migrando outliers para os pesos; depende de kernels INT8. *Uso:* discussão e limitação (não testável nativamente no Mac).
- **QLoRA:** voltado a *treinar* sobre pesos 4-bit. *Uso:* trabalhos futuros.
- **BitNet b1.58:** pesos ternários treinados do zero. *Uso:* limite inferior de bits (H3).
- **MCUNetV2:** o gargalo na borda é o pico de memória de ativações. *Uso:* justifica medir pico de memória e não só tamanho em disco.

## Diferenças
(a) surveys gerais × de quantização × de LLMs × de hardware; (b) pesos-só × pesos+ativações × treinados em baixo bit × ajuste fino; (c) quase todos avaliados em CUDA, e nenhum dos 13 foi mapeado como avaliação em Apple Silicon.

## Algoritmos e decisão
Ver tabela em [`../metodologia/vertente.md`](../metodologia/vertente.md). Usar: RTN, GPTQ, AWQ, DWQ (mlx-lm) e K-quants/imatrix (llama.cpp). Só referencial: SmoothQuant, QLoRA, BitNet, MCUNetV2. DWQ e K-quants **ainda não têm artigo de referência mapeado**.

---

# Rodada 2 (2026-10-22): trabalhos de 2025–2026

> **Base das descrições:** resumos do arXiv/Crossref (e, para #26 e #27, leitura direta do documento do repositório). **Nenhum PDF foi lido na íntegra**; afirmações sobre resultados são as dos próprios autores, não verificadas. Com exceção de #14 e #19, são preprints sem revisão por pares confirmada. Metadados e status em [`mapeamento.md`](mapeamento.md). Seção do artigo: **R** referencial, **M** método, **D** discussão, **F** trabalhos futuros.

## Apple Silicon / memória unificada

| # | Trabalho | Pontos relevantes | Diferença em relação aos anteriores | Entra em |
|---|---|---|---|---|
| 14 | Benazir & Lin 2026 (ACM SIGMETRICS PER) | Compara Apple Silicon e GPUs NVIDIA em latência/vazão; perfila ALU, largura de banda, cache; aponta custo de **dequantização**, compute e banda como gargalos | Primeira avaliação com revisão por pares de LLM em Apple Silicon no mapeamento; perspectiva de sistemas, não de acurácia | **R, D**; justifica medir banda/dequantização e interpretar tok/s no M5 Pro |
| 15 | Rajesh et al. 2025 | Compara 5 runtimes (MLX, MLC-LLM, llama.cpp, Ollama, PyTorch MPS) em M2 Ultra 192 GB; TTFT, vazão, contexto longo, suporte a quantização | Compara **runtimes**, não algoritmos de quantização; hardware muito maior que um MacBook | **R, M** (escolha de runtime), **D** |
| 16 | Javat & Kazakov 2026 | Nvidia Blackwell (NVFP4) × Apple Silicon para modelos 70B+; "VRAM wall" em GPU discreta | Foco em modelos grandes e em contraste CUDA×unificada; não é PTQ comparada | **D** (vantagem de capacidade da memória unificada) |
| 17 | Bergach 2026 (preprint, autor único) | Kernel Metal fundido de KV-cache int4 que, segundo o autor, supera fp16 em Apple Silicon | Única entrada de **KV-cache**; trata de uma técnica, não de survey; sem comparação com PTQ de pesos | **R** (KV-cache), **F**; cautela por ser autor único |
| 18 | Barrios 2026 (preprint, autor único) | vllm-mlx: servidor nativo em MLX, batching contínuo, 21–87% mais vazão que llama.cpp em M4 Max (segundo o resumo) | Sistema de serving, não quantização; mas dá referência de MLX×llama.cpp | **M**, **D** (comparação de runtimes), cautela |
| 21 | Chai et al. 2025 | Elasticidade de memória em dispositivos de memória unificada: conjunto de modelos quantizados trocáveis conforme a memória livre | Considera memória *compartilhada e flutuante* — conceito que os 13 anteriores não tratavam | **D, F** (memória como restrição dinâmica) |

## Fundamento dos métodos do experimento

| # | Trabalho | Pontos relevantes | Diferença | Entra em |
|---|---|---|---|---|
| 19 | Borgersen & Goodwin 2025 (Springer LNCS) | Explica K-quants + *importance matrix* e testa efeito do idioma da imatrix em desempenho multilingue de modelos GGUF | Único trabalho com veículo (Springer) sobre K-quants/imatrix que encontrei na busca (sem afirmar ineditismo); é sobre calibração por idioma, não compara com GPTQ/AWQ | **R** (K-quants/imatrix), **M** (escolha de dados da imatrix), **D** |
| 27 | Kawrakow 2023 (PR llama.cpp #1684 + README) | Introduz quants de 2–6 bits com *misturas* por tensor; `--imatrix`, `--pure`; perplexidade e KLD como métricas de perda | Documentação, não artigo; cita-se como fonte primária | **R, M** (reprodução: registrar commit) |
| 26 | MLX Team 2026 (LEARNED_QUANTS.md) | DWQ: ajusta parâmetros não quantizados (escalas, vieses) com o modelo original como professor; compara com AWQ, GPTQ e quantização dinâmica; funciona melhor em 2–4 bits | Documentação; **não há artigo de DWQ**; é o fundamento operacional | **R, M** |
| 23 | Xu et al. 2026 (QATFactory) | Framework de QAD/QARL; suporta NVFP4, MXFP4 e **Q4_K** do llama.cpp; exporta para vLLM e llama.cpp | Liga QAD aos formatos do experimento; GPU CUDA, não MLX | **R** (QAD como família de DWQ), **F** |
| 24 | Tu et al. 2026 | Mostra, via CKA, que QAD só com KL pode mascarar degradação interna; propõe preservar a geometria interna (NVFP4) | Diagnóstico da destilação para quantização; contexto NVFP4/RL-pós-treino | **R, D** (por que DWQ pode não bastar), **F** |
| 22 | Hu & Ramachandran 2026 | Em 9 modelos, o dano de quantização é **difuso** entre camadas; gastar o bit extra em granularidade fina global rende mais | Dá base empírica à ideia de bits mistos; não é K-quants; preprint em workshop | **R, D** |
| 20 | Egiazarian et al. 2025 | Estuda MXFP4/NVFP4 em PTQ; propõe MR-GPTQ; aponta limites do formato (escala de potência de 2, grupo pequeno) | Formatos de 4 bits com *microscaling*, suportados em GPUs NVIDIA/AMD, não em Apple Silicon | **R, D** (limitação), **F** |

## Benchmark de borda

| # | Trabalho | Pontos relevantes | Diferença | Entra em |
|---|---|---|---|---|
| 25 | Taherin et al. 2026 (Hydra, IISWC 2026 a conferir) | Caracterização por fase (prefill/decode) de 13 LLMs em 3 gerações Jetson (Xavier, Orin, Thor), 5 formatos de execução, com llama.cpp | Borda **NVIDIA Jetson**, não Apple; metodologia de medição por fase e artefato público | **M** (protocolo de medição), **D** |

## Resposta à lacuna "poucas avaliações em hardware de borda com memória unificada"

**Confirmada, mas atenuada.** A rodada 2 achou avaliações em Apple Silicon (#14–#16, #18) e um trabalho conceitual sobre memória unificada (#21), então a formulação "não há avaliações" deixou de valer: **há avaliações de desempenho/sistema**. O que **continua em aberto**, pelo que os resumos mostram: (i) nenhum compara os algoritmos de PTQ do projeto (RTN, GPTQ, AWQ, DWQ, K-quants/imatrix) **lado a lado em acurácia e desempenho no mesmo Mac**; (ii) as avaliações sólidas são de runtimes ou de sistemas (M2 Ultra, M4 Max), não de MacBook M5 Pro nem de LLMs pequenos; (iii) só #14 tem revisão por pares; (iv) os métodos novos de quantização/QAD (#20, #23, #24) são avaliados em CUDA. Sugestão de reformulação para o artigo: "avaliações **comparativas de algoritmos de quantização** em memória unificada, com acurácia e eficiência medidas juntas, ainda são escassas". Falta ler os textos completos para sustentar isso.

## Observações para falseabilidade (Popper)

- A afirmação de #17 ("int4 mais rápido que fp16 em Apple Silicon") e a de #14 (dequantização como gargalo) geram previsões testáveis no M5 Pro: p.ex., tok/s de 4 bits > 8 bits > fp16 em decodificação limitada por banda. Se não ocorrer, é refutação local.
- #19 prevê sensibilidade da imatrix à língua da calibração: testável trocando o corpus de calibração (relevante para H3).
