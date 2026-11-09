# Plano de Execução — Fechamento Analítico (Etapas 4b/5) e Redação Científica (Etapa 7)

**Data:** 2026-11-09  
**Spec:** [`spec.md`](file:///Users/andreatlas/Downloads/Edge%20AI%20&%20Quantization/docs/atlas/2026-11-09-fechamento-v3-analise-artigo/spec.md)  
**Ledger:** [`ledger.md`](file:///Users/andreatlas/Downloads/Edge%20AI%20&%20Quantization/docs/atlas/2026-11-09-fechamento-v3-analise-artigo/ledger.md)

---

## Tarefas por Lane

### Tarefa 1 · `[core]` Infraestrutura, Trava de Energia e Avaliação do `Qwen3-0.6B gptq_fp32 3b`
- **Arquivos:**
  - `experimentos/rodar_fim_de_semana.sh` (consolidar trava de tomada `AC Power`)
  - `experimentos/rodar_bloco1_v3.sh` (novo orquestrador focado apenas no Bloco 1 da `matriz-v3.json` — `gptq_fp32` nos 5 modelos da v2, com trava de tomada e backup automático)
  - `experimentos/resultados/resultados.csv` e `experimentos/resultados/nll/Qwen3-0.6B-gptq_fp32-3b-g64-s0.npy`
- **Ações:**
  1. Fazer backup de `experimentos/resultados/resultados.csv`.
  2. Executar `rodar.py` (fase `qualidade`) para o modelo já quantizado em disco `Qwen/Qwen3-0.6B` (`gptq_fp32`, 3 bits, grupo 64, seed 0), que detecta o diretório existente em `experimentos/modelos/Qwen3-0.6B-gptq_fp32-3b-g64-s0` e executa apenas a avaliação (WikiText-2 + 5 tarefas zero-shot).
  3. Verificar a condição de sanidade da seção 4.2 de `protocolo-v3.md`: $\text{PPL}_{\text{gptq\_fp32 4b}} (22{,}58) < \text{PPL}_{\text{gptq\_fp32 3b}} < \text{PPL}_{\text{rtn 3b}} (62{,}26)$.
  4. Criar `experimentos/rodar_bloco1_v3.sh` para executar os 8 jobs restantes do Bloco 1 (`Qwen3-1.7B`, `SmolLM3-3B`, `Llama-3.2-3B`, `Qwen3-4B` em 4 e 3 bits) com guarda de memória e trava de energia.

### Tarefa 2 · `[core]` Fechamento Analítico da Etapa 5 (`analise.py`, `testes.py`, `exportar_site.py`)
- **Arquivos:**
  - `experimentos/analise.py`
  - `experimentos/testes.py`
  - `experimentos/resultados/hipoteses.json`
  - `site/data/resultados.json`
- **Ações:**
  1. Escrever primeiro os testes em `experimentos/testes.py` (TDD) cobrindo:
     - `concordancia_acuracia(L)`: cálculo de $Q_{\text{acc}} = \text{acc\_media}_q / \text{acc\_media}_{\text{fp}}$, correlação de Spearman e Pearson entre $Q$ e $Q_{\text{acc}}$, e concordância do melhor método ciente em 4 bits.
     - `variancia_replicas_awq(L)`: cálculo de média, desvio-padrão amostral ($\sigma_{\text{cal}}$) e coeficiente de variação (CV%) para as sementes `0, 1, 2` em `Qwen3-0.6B` e `Qwen3-1.7B`.
  2. Implementar as duas funções diagnósticas em `experimentos/analise.py` e incluí-las na saída de `resultados/hipoteses.json` sob as chaves `"concordancia_q_acuracia"` e `"replicas_awq"`, mantendo inalterados os vereditos primários de H1–H4.
  3. Rodar `analise.py`, `analise_v3.py` e `exportar_site.py` para atualizar `hipoteses.json`, `hipoteses-v3.json` (parcial) e `site/data/resultados.json`.

### Tarefa 3 · `[core]` Elaboração das 27 Fichas de Leitura (`literatura/fichas/`)
- **Arquivos:**
  - `literatura/fichas/01-shuvo-2023.md` a `literatura/fichas/27-kawrakow-2023.md`
  - `literatura/fichas/README.md`
- **Ações:**
  1. Para cada um dos 27 trabalhos de `literatura/mapeamento.md` e `literatura/referencias.bib` (Rodada 1: `#01–#13`; Rodada 2: `#14–#27`), gerar a ficha completa seguindo `metodologia/ficha-modelo.md`:
     - Referência ABNT NBR 6023, chave BibTeX, DOI/arXiv e status de verificação (indicando explicitamente quando baseada no resumo verificado via API vs. leitura direta de documentação técnica, em conformidade com o rigor de transparência do projeto).
     - Seções 1 a 6: Análise textual, Análise temática, Análise interpretativa, Problematização para o nosso estudo no M5 Pro, Síntese pessoal e Falseabilidade (Popper).
  2. Criar `literatura/fichas/README.md` indexando as 27 fichas por categoria (A–E) e seção de destino no artigo (`R`, `M`, `D`, `F`).

### Tarefa 4 · `[arch]` Script Reprodutível de Tabelas/Figuras e Manuscrito Científico (`artigo/`)
- **Arquivos:**
  - `experimentos/gerar_tabelas_artigo.py`
  - `artigo/tabelas/*.md` e `artigo/tabelas/*.tex`
  - `artigo/figuras/*.svg`
  - `artigo/manuscrito.md`
  - `metodologia/processo.md`, `metodologia/caderno-de-laboratorio.md`, `README.md`
- **Ações:**
  1. Implementar `experimentos/gerar_tabelas_artigo.py` que lê `resultados/resultados.csv`, `resultados/kquant.csv`, `resultados/kquant-velocidade.csv`, `resultados/hipoteses.json` e `resultados/h3_nll.json` e gera automaticamente:
     - `tabela1_grade_modelos.md/.tex` (arquitetura, parâmetros, tamanho FP16/BF16, PPL e acurácia base dos 5 modelos)
     - `tabela2_qualidade_mlx.md/.tex` (matriz completa de $Q$, IC95% e $Q_{\text{acc}}$ para RTN, GPTQ-bf16, GPTQ-fp32, AWQ e DWQ em 8, 4, 3 e 2 bits)
     - `tabela3_velocidade_memoria_h4.md/.tex` (prefill tok/s, decode tok/s, pico de memória, tamanho em disco e eficiência $E$ em 4 bits — Rodadas 1 e 2)
     - `tabela4_kquants_llamacpp.md/.tex` (qualidade e velocidade de F16, Q8_0, Q4_K_M, Q3_K_M e Q2_K nos 5 modelos)
     - `tabela5_vereditos_hipoteses.md/.tex` (síntese de H1–H4 na v2, análise secundária em NLL, concordância $Q \times Q_{\text{acc}}$ e diagnóstico da Hessiana `bf16` vs `fp32`)
     - Figuras SVG vetoriais (`figura1_pareto_qualidade_tamanho.svg`, `figura2_degradacao_superlinear_nll.svg`, `figura3_eficiencia_velocidade_h4.svg`, `figura4_mlx_vs_kquants.svg`).
  2. Redigir `artigo/manuscrito.md` em português seguindo a estrutura Severino + Popper (`metodologia/estrutura-do-artigo.md`), integrando as 27 referências, os achados da Matriz v2, a demonstração algébrica/numérica do colapso por `NaN` na decomposição de Cholesky da Hessiana em `bf16` vs `fp32`, e a comparação entre `mlx-lm` e `llama.cpp`.
  3. Atualizar `metodologia/caderno-de-laboratorio.md` (Fase 25), `metodologia/processo.md` e `README.md`.
