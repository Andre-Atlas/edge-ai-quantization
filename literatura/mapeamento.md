# Mapeamento Bibliográfico — Edge AI & Quantization

Mapeamento inicial de 13 trabalhos (2021–2024) sobre *Efficient Deep Learning*, otimização de redes e inferência na borda. Entradas BibTeX em [`referencias.bib`](referencias.bib); matriz de síntese em [`matriz-de-sintese.csv`](matriz-de-sintese.csv).

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
