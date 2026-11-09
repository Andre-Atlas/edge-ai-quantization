# Índice Geral das Fichas de Leitura (Severino + Popper)

Este diretório contém as **27 fichas de leitura completas** que fundamentam o artigo científico do projeto **Edge AI & Quantization** (*Estudo experimental comparativo de quantização pós-treino de pequenos LLMs em Apple Silicon M5 Pro de 24 GiB*), cobrindo 100% das entradas de [`../referencias.bib`](../referencias.bib), [`../mapeamento.md`](../mapeamento.md) e [`../matriz-de-sintese.csv`](../matriz-de-sintese.csv).

## Metodologia e Transparência Científica

Todas as fichas seguem estritamente o modelo estrutural de **Antônio Joaquim Severino** (*Metodologia do Trabalho Científico* — veja [`../../metodologia/severino.md`](../../metodologia/severino.md) e [`../../metodologia/ficha-modelo.md`](../../metodologia/ficha-modelo.md)) combinado ao critério de demarcação e falseabilidade de **Karl Popper** (*A lógica da pesquisa científica* — veja [`../../metodologia/popper.md`](../../metodologia/popper.md)):

1. **Análise textual (visão global):** gênero acadêmico/técnico, estrutura argumentativa e vocabulário-chave.
2. **Análise temática:** tema, problema, tese/hipótese dos autores, método e resultados principais.
3. **Análise interpretativa:** contexto na literatura, pressupostos, pontos fortes e limitações.
4. **Problematização para o nosso estudo:** conexão direta com as hipóteses pré-registradas **H1, H2, H3 e H4** no Apple M5 Pro (24 GiB), a falha numérica da decomposição de Cholesky na Hessiana `bfloat16` do GPTQ padrão e sua solução em `float32` (`gptq_fp32`), a comparação entre `RTN`, `GPTQ`, `AWQ`, `DWQ` (`mlx-lm`) e `K-quants`/`imatrix` (`llama.cpp`).
5. **Síntese pessoal:** reelaboração autoral concisa do papel da obra no artigo.
6. **Falseabilidade (Popper):** condições empíricas de refutação da tese dos autores e limites de validade.
7. **Pontos-chave verificados para citação indireta:** síntese auditada sem fabricação de números de página ou citações literais entre aspas de PDFs não lidos na íntegra.

### Legenda das Seções de Uso no Artigo
- **`R`** — Referencial Teórico e Estado da Arte
- **`M`** — Metodologia e Protocolo Experimental
- **`D`** — Resultados e Discussão
- **`F`** — Limitações e Trabalhos Futuros

---

## Tabela-Índice das 27 Fichas (`#01` a `#27`)

| # | Arquivo | Chave BibTeX | Ano | Categoria | Status de Verificação | Seções de Uso no Artigo (`R`/`M`/`D`/`F`) |
|---|---|---|---|---|---|---|
| 01 | [`01-shuvo2023efficient.md`](01-shuvo2023efficient.md) | `shuvo2023efficient` | 2023 | A. Surveys e revisões | Crossref (DOI `10.1109/JPROC.2022.3226481` — *Proc. IEEE*); resumo verificado | `R`, `M`, `D` |
| 02 | [`02-menghani2023efficient.md`](02-menghani2023efficient.md) | `menghani2023efficient` | 2023 | A. Surveys e revisões | Crossref (DOI `10.1145/3578938` — *ACM CSUR*); resumo verificado | `R` |
| 03 | [`03-dantas2024comprehensive.md`](03-dantas2024comprehensive.md) | `dantas2024comprehensive` | 2024 | A. Surveys e revisões | Crossref (DOI `10.1007/s10489-024-05747-w` — *Applied Intelligence*); resumo verificado | `R` |
| 04 | [`04-lin2023tiny.md`](04-lin2023tiny.md) | `lin2023tiny` | 2023 | A. Surveys e revisões | Crossref (DOI `10.1109/MCAS.2023.3302182` — *IEEE CAS Mag.*) + arXiv (`2403.19076`); resumo verificado | `R`, `F` |
| 05 | [`05-zhu2024survey.md`](05-zhu2024survey.md) | `zhu2024survey` | 2024 | A. Surveys e revisões | Crossref (DOI `10.1162/tacl_a_00704` — *TACL*); resumo verificado | `R`, `D` |
| 06 | [`06-gholami2022survey.md`](06-gholami2022survey.md) | `gholami2022survey` | 2022 | A. Surveys e revisões | arXiv (`2103.13630`, cap. *Low-Power Computer Vision* CRC); resumo verificado | `R`, `M` |
| 07 | [`07-nagel2021white.md`](07-nagel2021white.md) | `nagel2021white` | 2021 | A. Surveys e revisões | arXiv (`2106.08295` — *Qualcomm AI Research White Paper*); resumo verificado | `R`, `M`, `D` |
| 08 | [`08-frantar2023gptq.md`](08-frantar2023gptq.md) | `frantar2023gptq` | 2023 | B. Algoritmos e métodos | arXiv (`2210.17323`, ICLR 2023); resumo e formulação da Hessiana/Cholesky verificados | `R`, `M`, `D` |
| 09 | [`09-xiao2023smoothquant.md`](09-xiao2023smoothquant.md) | `xiao2023smoothquant` | 2023 | B. Algoritmos e métodos | arXiv (`2211.10438`, ICML 2023); resumo verificado | `R`, `D`, `F` |
| 10 | [`10-lin2024awq.md`](10-lin2024awq.md) | `lin2024awq` | 2024 | B. Algoritmos e métodos | arXiv (`2306.00978`, MLSys 2024); resumo verificado | `R`, `M`, `D` |
| 11 | [`11-dettmers2023qlora.md`](11-dettmers2023qlora.md) | `dettmers2023qlora` | 2023 | B. Algoritmos e métodos | arXiv (`2305.14314`, NeurIPS 2023); resumo verificado | `R`, `F` |
| 12 | [`12-ma2024era.md`](12-ma2024era.md) | `ma2024era` | 2024 | B. Algoritmos e métodos | arXiv (`2402.17764` — *BitNet b1.58*); resumo verificado | `R`, `D`, `F` |
| 13 | [`13-lin2021mcunetv2.md`](13-lin2021mcunetv2.md) | `lin2021mcunetv2` | 2021 | B. Algoritmos e métodos | arXiv (`2110.15352`, NeurIPS 2021); resumo verificado | `R`, `M` |
| 14 | [`14-benazir2026benchmarking.md`](14-benazir2026benchmarking.md) | `benazir2026benchmarking` | 2026 | C. Apple Silicon, MLX e memória unificada | Crossref (DOI `10.1145/3820171.3806865` — *ACM SIGMETRICS PER*) + arXiv (`2508.08531`); resumo verificado | `R`, `D` |
| 15 | [`15-rajesh2025production.md`](15-rajesh2025production.md) | `rajesh2025production` | 2025 | C. Apple Silicon, MLX e memória unificada | arXiv (`2511.05502`, preprint); resumo verificado | `R`, `M`, `D` |
| 16 | [`16-javat2026silicon.md`](16-javat2026silicon.md) | `javat2026silicon` | 2026 | C. Apple Silicon, MLX e memória unificada | arXiv (`2605.00519` v2, preprint); resumo verificado | `D` |
| 17 | [`17-bergach2026quantization.md`](17-bergach2026quantization.md) | `bergach2026quantization` | 2026 | C. Apple Silicon, MLX e memória unificada | arXiv (`2605.05699`, preprint de autor único); resumo verificado | `R`, `F` |
| 18 | [`18-barrios2026native.md`](18-barrios2026native.md) | `barrios2026native` | 2026 | C. Apple Silicon, MLX e memória unificada | arXiv (`2601.19139` v2, preprint de autor único); resumo verificado | `M`, `D` |
| 19 | [`19-borgersen2025english.md`](19-borgersen2025english.md) | `borgersen2025english` | 2025 | D. Métodos de borda, bits mistos e formatos | Crossref (DOI `10.1007/978-3-032-11402-0_8` — *Springer LNCS*) + arXiv (`2503.03592`); resumo verificado | `R`, `M`, `D` |
| 20 | [`20-egiazarian2025bridging.md`](20-egiazarian2025bridging.md) | `egiazarian2025bridging` | 2025 | D. Métodos de borda, bits mistos e formatos | arXiv (`2509.23202` v3, preprint); resumo verificado | `R`, `D`, `F` |
| 21 | [`21-chai2025flexquant.md`](21-chai2025flexquant.md) | `chai2025flexquant` | 2025 | D. Métodos de borda, bits mistos e formatos | arXiv (`2501.07139` v2, preprint); resumo verificado | `D`, `F` |
| 22 | [`22-hu2026structure.md`](22-hu2026structure.md) | `hu2026structure` | 2026 | D. Métodos de borda, bits mistos e formatos | arXiv (`2609.01587`, preprint sob revisão em workshop NeurIPS 2026); resumo verificado | `R`, `D` |
| 23 | [`23-xu2026qatfactory.md`](23-xu2026qatfactory.md) | `xu2026qatfactory` | 2026 | D. Métodos de borda, bits mistos e formatos | arXiv (`2609.39223` v2, preprint); resumo verificado | `R`, `F` |
| 24 | [`24-tu2026beyond.md`](24-tu2026beyond.md) | `tu2026beyond` | 2026 | D. Métodos de borda, bits mistos e formatos | arXiv (`2606.05682` v2, preprint); resumo verificado | `R`, `D`, `F` |
| 25 | [`25-taherin2026hydra.md`](25-taherin2026hydra.md) | `taherin2026hydra` | 2026 | E. Benchmark de borda e documentação técnica | arXiv (`2608.25053`, aceito no IISWC 2026 segundo arXiv); resumo verificado | `M`, `D` |
| 26 | [`26-mlxlm2026dwq.md`](26-mlxlm2026dwq.md) | `mlxlm2026dwq` | 2026 | E. Benchmark de borda e documentação técnica | Documentação técnica oficial lida na íntegra (`ml-explore/mlx-lm`, `LEARNED_QUANTS.md` + código `0.32.2`) | `R`, `M`, `D` |
| 27 | [`27-kawrakow2023kquants.md`](27-kawrakow2023kquants.md) | `kawrakow2023kquants` | 2023 | E. Benchmark de borda e documentação técnica | Documentação técnica oficial e PR `#1684` lidos diretamente (`ggml-org/llama.cpp`, `tools/quantize/README.md`) | `R`, `M`, `D` |

---

## Distribuição por Categoria Temática

- **Categoria A — Surveys e revisões de base (`#01` a `#07`):** 7 fichas (escopo macro de *Efficient Deep Learning*, inferência na borda, TinyML e fundamentos matemáticos de PTQ/QAT).
- **Categoria B — Algoritmos clássicos de otimização e quantização (`#08` a `#13`):** 6 fichas (`GPTQ`, `SmoothQuant`, `AWQ`, `QLoRA`, `BitNet b1.58`, `MCUNetV2`).
- **Categoria C — Apple Silicon, MLX e memória unificada (`#14` a `#18`):** 5 fichas (perfilamento microarquitetural, comparação de runtimes `MLX` vs. `llama.cpp`, barreiras de ecossistema e KV-cache `int4` em Metal).
- **Categoria D — Métodos de borda, bits mistos, formatos e destilação (`#19` a `#24`):** 6 fichas (`K-quants`/`imatrix` multilíngue, `MXFP4`/`NVFP4` e `MR-GPTQ`, elasticidade de memória `FlexQuant`, dano difuso de quantização, `QATFactory` e preservação da geometria interna na destilação).
- **Categoria E — Benchmark de borda por fase e documentação técnica primária (`#25` a `#27`):** 3 fichas (benchmark *phase-aware* `Hydra`, documentação primária do `DWQ`/`AWQ`/`GPTQ` no `mlx-lm` e especificação original dos `K-quants` no `llama.cpp`).
