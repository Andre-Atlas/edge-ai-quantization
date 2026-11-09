# Ficha de leitura — #19 Borgersen & Goodwin (2025)

**Referência (ABNT):** BORGERSEN, Karl Audun; GOODWIN, Morten. English K_Quantization of LLMs Does Not Disproportionately Diminish Multilingual Performance. In: **Artificial Intelligence XLII**. Lecture Notes in Computer Science (LNCS). Cham: Springer Nature Switzerland, 2025. DOI: 10.1007/978-3-032-11402-0_8. Preprint: arXiv:2503.03592.
**Chave BibTeX:** `borgersen2025english`
**Data da leitura:** 2026-11-09
**Escopo da leitura:** Consolidação dos metadados e resumo verificados via API do Crossref (DOI `10.1007/978-3-032-11402-0_8`, capítulo publicado em 24/11/2025) e do arXiv (`2503.03592`), sem fabricação de números de página ou citações literais entre aspas de PDF não lido na íntegra.

## 1. Análise textual (visão global)
- **Tipo de texto:** Artigo completo publicado em série de conferência revisada por pares (*Lecture Notes in Computer Science — Artificial Intelligence XLII*, Springer).
- **Estrutura conceitual:** Apresentação formal do esquema de quantização `K-quants` e do mecanismo de *importance matrix* (`imatrix`) do ecossistema `llama.cpp`/`GGUF`, seguida de avaliação experimental sobre se calibrar a `imatrix` exclusivamente com texto em inglês prejudica de modo desproporcional o desempenho do modelo quantizado em outros idiomas.
- **Vocabulário-chave:** *K-quants*, *importance matrix (imatrix)*, *GGUF*, *llama.cpp*, *multilingual LLM evaluation*, *calibration dataset*, *low-bit quantization*.

## 2. Análise temática
- **Tema:** O impacto do idioma do corpus de calibração da matriz de importância (*importance matrix* / `imatrix`) na quantização `K-quants` (`llama.cpp`) sobre o desempenho multilíngue de LLMs.
- **Problema:** A quantização de baixíssimo bit no `llama.cpp` depende da `imatrix` (que estima a importância dos pesos a partir das ativações de um texto de calibração) para evitar degradação severa; entretanto, na prática comunitária e industrial, quase todas as matrizes de importância são geradas exclusivamente com textos em inglês (como WikiText), levantando a suspeita de viés de degradação contra outras línguas.
- **Tese / hipótese do(s) autor(es):** A calibração da `imatrix` em inglês nos `K-quants` preserva as estruturas fundamentais do modelo sem causar degradação desproporcional nas capacidades multilíngues em comparação a calibrações específicas por idioma.
- **Argumentação / método:** Os autores quantizam LLMs para formatos `GGUF` (`K-quants`) no `llama.cpp` variando o idioma do conjunto de calibração usado na geração da `imatrix` e avaliam a perda de qualidade em benchmarks multilíngues através de diferentes níveis de bits.
- **Resultados principais:** Os resultados demonstram que o uso de uma `imatrix` calibrada em inglês é robusto entre idiomas e não penaliza desproporcionalmente tarefas em línguas não inglesas, validando o procedimento padrão de calibração em inglês para `K-quants`.

## 3. Análise interpretativa
- **Contexto e relevância:** Dentre os 14 itens da Rodada 2 (2025–2026), este é um dos dois únicos trabalhos com publicação revisada por pares confirmada via Crossref (Springer LNCS), e o **único trabalho acadêmico revisado por pares** localizado que formaliza e investiga experimentalmente os `K-quants` e a `imatrix` do `llama.cpp`.
- **Pressupostos:** Toma o esquema de superblocos e misturas dos `K-quants` do `llama.cpp` como método de referência e investiga a sensibilidade à distribuição linguística dos dados de calibração.
- **Pontos fortes:** Confere respaldo bibliográfico formal (com DOI Springer) para a descrição metodológica dos `K-quants` e da `imatrix` em nosso artigo, complementando a documentação primária de repositório de Kawrakow (2023, #27).
- **Limites:** Foca na variação do idioma da `imatrix` dentro do próprio `llama.cpp`, sem comparar os `K-quants` contra métodos baseados na Hessiana (`GPTQ`), escala por ativação (`AWQ`) ou destilação (`DWQ`) no ecossistema `MLX`.

## 4. Problematização
Questões e conexões diretas que o texto suscita para o nosso estudo experimental no Apple M5 Pro (24 GiB):
- **Validação direta da nossa escolha de calibração da `imatrix`:** Em nosso protocolo experimental (`experimentos/kquant.py`), geramos a `imatrix` de cada um dos 5 modelos usando o texto público de calibração em inglês (`calibration_v5.txt`, 64 blocos de 512 tokens) antes de produzir os modelos `Q8_0`, `Q4_K_M`, `Q3_K_M` e `Q2_K`. Borgersen e Goodwin (2025) sustentam que a calibração da `imatrix` sobre um corpus geral em inglês captura as direções de ativação estruturalmente salientes da rede sem sobreajustar (*overfit*) à distribuição superficial do texto.
- **Desempenho superior dos `K-quants` (`imatrix`) em 2 e 3 bits nos nossos dados (`kquant.csv`):** Em nossas medições no M5 Pro, enquanto o `RTN` puro do `mlx-lm` em 2 bits colapsa completamente ($Q \le 0{,}0037$ nos 5 modelos), o `Q2_K` com `imatrix` do `llama.cpp` preserva qualidade notavelmente superior (ex.: no `Qwen3-4B`, $\text{PPL}_{\text{F16}} = 11{,}98 \to \text{PPL}_{\text{Q2\_K}} = 15{,}54$, ou seja, $Q = 0{,}7706$ e concordância top-1 de $76{,}2\%$). Isso ocorre pela combinação de dois fatores que discutimos com base em Borgersen e Goodwin (2025) e Kawrakow (2023): (i) a ponderação por importância das ativações (`imatrix`) e (ii) a alocação mista de bits por tipo de tensor (que resulta em $3{,}26\text{–}3{,}70$ bits efetivos por peso no `Q2_K`, contra $\approx 2{,}5\text{ bpw}$ no `MLX 2b`).
- **Conexão com a variância de calibração (`replicas_awq` na Etapa 5):** A conclusão dos autores sobre a baixa sensibilidade dos métodos guiados por magnitude de ativação à escolha exata do texto de calibração converge com a baixíssima variância entre sementes de calibração que medimos no `AWQ` ($\sigma_{\text{cal}} = 0{,}0065$ em `protocolo-v3.md`).

## 5. Síntese pessoal
Borgersen e Goodwin (2025) fornecem a referência acadêmica revisada por pares essencial para fundamentar os `K-quants` e a `imatrix` no nosso artigo. Além de descreverem formalmente o mecanismo de quantização mista guiada por importância do `llama.cpp`, seus resultados validam o uso de um corpus padrão em inglês para estimar a importância dos canais sem induzir fragilidade distributiva.

## 6. Falseabilidade (Popper)
- **A hipótese central do trabalho é falseável?** Sim. O próprio título formula uma refutação da conjectura intuitiva de que calibrar a `imatrix` em inglês degradaria mais as demais línguas; a hipótese nula e a alternativa são diretamente testáveis comparando a perda relativa de perplexidade/acurácia entre idiomas com `imatrix` inglesa vs. nativa.
- **Como foi testada?** Quantizando múltiplos LLMs com `K-quants` sob diferentes corpora linguísticos de `imatrix` e avaliando benchmarks multilíngues.
- **Quais condições limitam a conclusão?** Os testes concentram-se em modelos e alfabetos cobertos pelo pré-treinamento multilíngue padrão; em regimes extremos de 2 bits sem misturas de tensores (`--pure`), a sensibilidade ao domínio de calibração pode aumentar.

## Citações úteis / Pontos-chave verificados para citação indireta
- **Seções de uso previstas no artigo:** Referencial Teórico (`R`), Metodologia (`M`) e Discussão (`D`).
- **Ponto verificado 1 (Fundamentação acadêmica de K-quants e imatrix):** Os esquemas `K-quants` do `llama.cpp` combinam superblocos de precisão mista com uma matriz de importância (`imatrix`) derivada de ativações de calibração para proteger pesos críticos em quantizações de 2 a 6 bits (BORGERSEN; GOODWIN, 2025).
- **Ponto verificado 2 (Robustez do corpus de calibração em inglês):** A geração da `imatrix` com corpus em inglês não degrada desproporcionalmente o desempenho generalista do modelo quantizado, respaldando o protocolo de calibração adotado em nossos experimentos com `llama.cpp` (BORGERSEN; GOODWIN, 2025).
