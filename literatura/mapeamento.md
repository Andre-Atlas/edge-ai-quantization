# Mapeamento Bibliográfico — Edge AI & Quantization

Mapeamento inicial de 13 trabalhos (2021–2024) sobre *Efficient Deep Learning*, otimização de redes e inferência na borda. Entradas BibTeX em [`referencias.bib`](referencias.bib); matriz de síntese em [`matriz-de-sintese.csv`](matriz-de-sintese.csv). **Atualização:** a rodada 2 (2026-10-22, 2025–2026) está na seção final deste arquivo.

> **Status de verificação.** Título, autores, ano e DOI foram conferidos em Crossref (artigos de periódico) ou na API do arXiv (preprints). Para os trabalhos de conferência (GPTQ, SmoothQuant, AWQ, QLoRA, MCUNetV2), o veículo indicado vem de conhecimento prévio e **não** foi confirmado por API: conferir antes de citar. As descrições abaixo vêm de resumos/títulos; **nenhum artigo foi lido na íntegra ainda**. A leitura completa é a próxima etapa (ver Severino em [`../metodologia/severino.md`](../metodologia/severino.md)).

## A. Surveys e revisões (base do estado da arte)

| # | Referência | Veículo / base | Por que importa |
|---|---|---|---|
| 1 | Shuvo et al. (2023). *Efficient Acceleration of Deep Learning Inference on Resource-Constrained Edge Devices: A Review.* [10.1109/JPROC.2022.3226481](https://doi.org/10.1109/JPROC.2022.3226481) | Proceedings of the IEEE — **IEEE Xplore** | Panorama de inferência na borda: arquitetura, algoritmos, co-projeto HW/SW e aceleradores. |
| 2 | Menghani (2023). *Efficient Deep Learning: A Survey on Making Deep Learning Models Smaller, Faster, and Better.* [10.1145/3578938](https://doi.org/10.1145/3578938) | ACM Computing Surveys — **ACM DL** | Taxonomia central de *Efficient Deep Learning*. |
| 3 | Dantas et al. (2024). *A comprehensive review of model compression techniques in machine learning.* [10.1007/s10489-024-05747-w](https://doi.org/10.1007/s10489-024-05747-w) | Applied Intelligence — **Springer** | Revisão de poda, quantização, destilação etc. |
| 4 | Lin et al. (2023). *Tiny Machine Learning: Progress and Futures.* [10.1109/MCAS.2023.3302182](https://doi.org/10.1109/MCAS.2023.3302182) | IEEE Circuits and Systems Magazine — **IEEE Xplore** | TinyML e co-projeto algoritmo–sistema em microcontroladores. |
| 5 | Zhu et al. (2024). *A Survey on Model Compression for Large Language Models.* [10.1162/tacl_a_00704](https://doi.org/10.1162/tacl_a_00704) | TACL | Compressão específica para LLMs. |
| 6 | Gholami et al. (2022). *A Survey of Quantization Methods for Efficient Neural Network Inference.* [arXiv:2103.13630](https://arxiv.org/abs/2103.13630) | Capítulo de livro (Chapman & Hall/CRC); preprint 2021 | Survey de referência sobre quantização. |
| 7 | Nagel et al. (2021). *A White Paper on Neural Network Quantization.* [arXiv:2106.08295](https://arxiv.org/abs/2106.08295) | arXiv (Qualcomm AI Research) | Guia prático de PTQ e QAT. |

## B. Algoritmos e métodos de otimização

| # | Referência | Contribuição (segundo o resumo) |
|---|---|---|
| 8 | Frantar et al. (2023). *GPTQ.* [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) | PTQ one-shot de pesos de LLMs a 3–4 bits. |
| 9 | Xiao et al. (2023). *SmoothQuant.* [arXiv:2211.10438](https://arxiv.org/abs/2211.10438) | PTQ W8A8: transfere a dificuldade das ativações para os pesos. |
| 10 | Lin et al. (2024). *AWQ.* [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) | Quantização só-pesos guiada por ativações; runtime voltado a GPUs de borda. |
| 11 | Dettmers et al. (2023). *QLoRA.* [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) | Fine-tuning sobre pesos de 4 bits. |
| 12 | Ma et al. (2024). *The Era of 1-bit LLMs (1.58 bits).* [arXiv:2402.17764](https://arxiv.org/abs/2402.17764) | Pesos ternários; limite inferior de precisão. |
| 13 | Lin et al. (2021). *MCUNetV2.* [arXiv:2110.15352](https://arxiv.org/abs/2110.15352) | Inferência por *patches* para reduzir pico de memória em MCUs. |

## Cobertura por critério da demanda

- **Bases:** IEEE Xplore (#1, #4), ACM DL (#2), Springer (#3); demais em TACL/arXiv/conferências.
- **Janela temporal:** 2021–2024 (últimos 3–5 anos).
- **Lacuna conhecida:** nada de 2025–2026 ainda, e poucos trabalhos de hardware/aceleradores e de *edge* além de LLMs/MCUs. Próxima rodada de busca deve cobrir 2025–2026, QAT, quantização de modelos de visão/ViT, NPUs e benchmarks de borda (ex.: MLPerf Tiny).

## Estratégia de busca (para reprodutibilidade, cf. Severino)

A ser registrada em cada rodada: base, *string* de busca, data, filtros, nº de resultados, critérios de inclusão/exclusão. Rodada 1 (2026-10-22): buscas livres por *efficient deep learning*, *quantization*, *edge inference*, *TinyML*, *LLM post-training quantization*; seleção por relevância, citação/reconhecimento e diversidade de bases.

---

## Rodada 2 (2026-10-22)

Foco: 2025–2026. 12 trabalhos + 2 documentos técnicos (DWQ e K-quants/imatrix, que não têm artigo). Entradas em [`referencias.bib`](referencias.bib) e [`matriz-de-sintese.csv`](matriz-de-sintese.csv); análise em [`analise-dos-artigos.md`](analise-dos-artigos.md).

> **Status de verificação.** Todas as entradas foram buscadas na API do arXiv (título, autores, data, comentários) e, quando há DOI/versão publicada, no Crossref. Só **2** artigos têm veículo confirmado por Crossref: #14 (ACM SIGMETRICS PER) e #19 (LNCS/Springer); #26 e #27 são documentos de repositório que li diretamente. Os demais são **preprints do arXiv sem revisão por pares confirmada**; marcados "a conferir". As descrições vêm dos **resumos** (DWQ/K-quants: leitura direta do documento); **nenhum PDF foi lido na íntegra**. Números de volume/páginas e editores ficaram "a conferir".

### Bases, strings e critérios

- **Bases efetivamente consultadas:** arXiv (API, ordenada por data) e Crossref (API; busca por DOI e por `query.bibliographic`); GitHub (API/raw) para documentação técnica. **IEEE Xplore, ACM DL e Springer Link não foram consultados diretamente** (sem acesso por API nesta sessão); ACM DL e Springer entram só via Crossref (DOIs #14 e #19). Conferências (MLSys, ICLR, NeurIPS, ICML, ACL) não foram buscadas por site próprio: apareceram apenas se o arXiv/Crossref indicasse o veículo, e nenhum indicou.
- **Strings (arXiv, campos `all:`, filtro 2025–2026):** `"Apple Silicon" AND quantization`; `MLX AND LLM AND inference`; `"unified memory" AND LLM AND quantization`; `"KV cache" AND quantization AND (survey|edge)`; `"KV cache quantization" AND evaluation`; `MXFP4 AND quantization`; `NVFP4`; `microscaling AND FP4 AND LLM`; `"llama.cpp" AND quantization`; `GGUF AND quantization`; `"importance matrix" AND quantization`; `distillation AND "weight quantization" AND LLM`; `"mixed-precision" AND LLM AND "post-training quantization"`; `edge AND benchmark AND quantized AND LLM`.
- **Inclusão:** (i) ano 2025–2026; (ii) quantização de LLMs/small LMs em hardware de consumo/borda, ou fundamento direto de um dos métodos do experimento (DWQ, K-quants/imatrix, MXFP4); (iii) metadados confirmáveis por API; (iv) cobertura de cada tema pedido.
- **Exclusão:** (i) foco em treino/pré-treino (ex.: NVFP4 *pretraining*) ou em hardware dedicado/aceleradores; (ii) só GPU de datacenter sem relação com a lacuna; (iii) resultados sobre sistemas de serving sem quantização como variável; (iv) preferência por trabalhos com mais de um autor/veículo sempre que equivalentes — mas mantive preprints de autor único quando eram o único trabalho de Apple Silicon/KV-cache (#17, #18), sinalizados.
- **Limitações:** busca por ordem de data (não por citação), pois trabalhos de 2026 ainda têm poucas citações; não houve snowballing; **KV-cache**: não achei survey de 2025–2026 por API — só estudo de uma técnica (#17); candidatos vistos mas não incluídos: Khatib et al. (WIMS 2026, arXiv:2609.08307, Jetson/LLM), Gautam & Jha (RAMP, arXiv:2603.17891, bits mistos por RL), Singh et al. (arXiv:2610.04403, GGUF a Q2_K em SLMs).

### C. Apple Silicon, MLX e memória unificada

| # | Referência | Veículo / verificação |
|---|---|---|
| 14 | Benazir & Lin (2026). *Benchmarking and Characterization of LLM Inference on Apple Silicon.* [10.1145/3820171.3806865](https://doi.org/10.1145/3820171.3806865) (preprint [arXiv:2508.08531](https://arxiv.org/abs/2508.08531), *Profiling LLM Inference on Apple Silicon: A Quantization Perspective*) | ACM SIGMETRICS PER — **ACM DL**; Crossref ✔ (título do preprint difere; abstracts coincidem) |
| 15 | Rajesh et al. (2025). *Production-Grade Local LLM Inference on Apple Silicon: MLX, MLC-LLM, Ollama, llama.cpp, PyTorch MPS.* [arXiv:2511.05502](https://arxiv.org/abs/2511.05502) | arXiv ✔; preprint |
| 16 | Javat & Kazakov (2026). *Silicon Showdown: … Consumer-Grade LLM Inference.* [arXiv:2605.00519](https://arxiv.org/abs/2605.00519) | arXiv ✔; preprint |
| 17 | Bergach (2026). *When Quantization Is Free: An int4 KV Cache That Outruns fp16 on Apple Silicon.* [arXiv:2605.05699](https://arxiv.org/abs/2605.05699) | arXiv ✔; preprint de autor único (KV-cache) |
| 18 | Barrios (2026). *Native LLM and MLLM Inference at Scale on Apple Silicon (vllm-mlx).* [arXiv:2601.19139](https://arxiv.org/abs/2601.19139) | arXiv ✔; preprint de autor único |

### D. Métodos de borda, bits mistos e formatos

| # | Referência | Veículo / verificação |
|---|---|---|
| 19 | Borgersen & Goodwin (2025). *English K_Quantization of LLMs Does Not Disproportionately Diminish Multilingual Performance.* [10.1007/978-3-032-11402-0_8](https://doi.org/10.1007/978-3-032-11402-0_8) ([arXiv:2503.03592](https://arxiv.org/abs/2503.03592)) | LNCS (*Artificial Intelligence XLII*) — **Springer**; Crossref ✔ |
| 20 | Egiazarian et al. (2025). *Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization.* [arXiv:2509.23202](https://arxiv.org/abs/2509.23202) | arXiv ✔; veículo a conferir |
| 21 | Chai et al. (2025). *FlexQuant: Elastic Quantization Framework for Locally Hosted LLM on Edge Devices.* [arXiv:2501.07139](https://arxiv.org/abs/2501.07139) | arXiv ✔; veículo a conferir (outro "FlexQuant" no EMNLP Findings 2025 tem título diferente) |
| 22 | Hu & Ramachandran (2026). *The Structure of Quantization Damage in LLMs: Why the Next Bit Should Be Spent Globally.* [arXiv:2609.01587](https://arxiv.org/abs/2609.01587) | arXiv ✔; sob revisão em workshop |
| 23 | Xu et al. (2026). *QATFactory: … Quantization-aware Training and Distillation of LLMs.* [arXiv:2609.39223](https://arxiv.org/abs/2609.39223) | arXiv ✔; preprint |
| 24 | Tu et al. (2026). *Beyond Output Matching: Preserving Internal Geometry in NVFP4 LLM Distillation.* [arXiv:2606.05682](https://arxiv.org/abs/2606.05682) | arXiv ✔; preprint |

### E. Benchmark de borda e documentação técnica

| # | Referência | Veículo / verificação |
|---|---|---|
| 25 | Taherin et al. (2026). *Hydra: Phase-Aware Workload Characterization of LLM Inference across Edge SoC Generations, Backends, and Quantization Levels.* [arXiv:2608.25053](https://arxiv.org/abs/2608.25053) | arXiv ✔; "aceito no IISWC 2026" só pelo comentário do arXiv (a conferir) |
| 26 | MLX Team (2026). *Learned Quantization (DWQ, AWQ, dynamic, GPTQ).* [mlx-lm/LEARNED_QUANTS.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LEARNED_QUANTS.md) | Documentação (MIT); lida; acesso 2026-10-22 |
| 27 | Kawrakow (2023). *k-quants* (PR [#1684](https://github.com/ggml-org/llama.cpp/pull/1684)) e [README do `llama-quantize`](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md) | Documentação/PR; trechos iniciais lidos; acesso 2026-10-22 |

(Numeração continua a da rodada 1: #14–#25 são os 12 trabalhos acadêmicos e #26–#27 os 2 documentos técnicos.)

### Lacunas que permanecem após a rodada 2

1. Sem avaliação **peer-reviewed** de quantização em **MLX** (#14 é a única com veículo e usa Apple Silicon sem foco em MLX/DWQ; #15/#18 são preprints sobre runtimes).
2. **Nenhum** trabalho encontrado compara RTN × GPTQ × AWQ × DWQ × K-quants no mesmo hardware de memória unificada.
3. DWQ e K-quants/imatrix seguem **sem artigo**: só documentação técnica (#26, #27); a base teórica de DWQ vem de destilação/QAD (#23, #24, ambos em NVFP4/GPU, não MLX).
4. **KV-cache:** sem survey 2025–2026 verificado; só um preprint de autor único em Apple Silicon (#17).
5. Dos itens, só #14 e #19 têm confirmação de veículo; o resto exige conferir se saiu versão publicada antes da submissão.
6. IEEE Xplore, ACM DL (busca direta), Springer Link e sites de MLSys/ICLR/NeurIPS/ICML/ACL não foram pesquisados diretamente.
