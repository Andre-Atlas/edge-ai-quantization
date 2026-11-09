# Especificação (Spec) — Fechamento Analítico (Etapas 4b/5) e Redação Científica (Etapa 7)

**Data:** 2026-11-09  
**Linha ATLAS:** `ATLAS · feature · arquitetura+dados+produto · large → pesquisa, descoberta, plano, G1, SDD, revisão, verificação, G2 · lanes: arquitetura`

---

## 1. Contexto e Estado Atual Auditado

1. **Repositório e Workspace:**
   - Repositório remoto `https://github.com/Andre-Atlas/edge-ai-quantization` (`main` em `3691822`) sincronizado localmente em `/Users/andreatlas/Downloads/Edge AI & Quantization` (migrado do `scratch-workspaces` preservando os 163,5 GB de modelos quantizados, GGUFs/`llama.cpp`, logs e `.venv`).
2. **Estado Experimental:**
   - **Matriz v2 (Etapas 0–4):** Concluída com 68 execuções de qualidade (59 válidas + 9 inválidas por `NaN` no GPTQ `bf16` nos modelos de 3–4B) e 2 rodadas de velocidade (59 configurações MLX + 25 K-quants `llama.cpp`).
   - **Vereditos v2 (`resultados/hipoteses.json`):** H4 corroborada ($E = 0{,}83\text{–}0{,}87 \ge 0{,}60$); H1, H2 e H3 inconclusivas pelos limiares absolutos da v1 (com 1ª metade da H3 superlinear nos 5 modelos na escala secundária pós-hoc em NLL).
   - **Extensão Pré-Registrada (Etapa 4b / `protocolo-v3`):**
     - `Qwen/Qwen3-0.6B gptq_fp32 4b` concluído (`PPL = 22,58057`, $Q = 0{,}9287$).
     - `Qwen/Qwen3-0.6B gptq_fp32 3b` já quantizado em disco (`experimentos/modelos/Qwen3-0.6B-gptq_fp32-3b-g64-s0`), pendente apenas da avaliação de perplexidade e zero-shot.

---

## 2. Escopo Definido pelo Pesquisador

1. **Etapa 4b (Escopo Focado nos 5 Modelos da v2 — Bloco 1 de `matriz-v3.json`):**
   - Concluir imediatamente a avaliação do `Qwen3-0.6B gptq_fp32 3b` já quantizado em disco (fechando a validação de sanidade da seção 4 do `protocolo-v3.md`).
   - Executar apenas o bloco 1 da `matriz-v3.json` (`gptq_fp32` em 4 e 3 bits nos 5 modelos da v2: `Qwen3-0.6B`, `Qwen3-1.7B`, `SmolLM3-3B`, `Llama-3.2-3B`, `Qwen3-4B` — restando 8 quantizações + 1 avaliação já quantizada) para comprovar empiricamente que a Hessiana em `float32` elimina os `NaN` nos modelos de 3–4B e medir seu impacto sem atrasar o cronograma de 18/12/2026.
   - Manter a trava de energia (`AC Power`) em `experimentos/rodar_fim_de_semana.sh` e criar um script dedicado `experimentos/rodar_bloco1_v3.sh` para rodar/retomar com segurança apenas o bloco 1 da v3.
2. **Etapa 5 (Fechamento Analítico em `experimentos/analise.py` e `analise_v3.py`):**
   - Preservar integralmente os critérios primários pré-registrados (`protocolo-v2` e `protocolo-v3`).
   - Adicionar ao `analise.py` (e `hipoteses.json`) as duas análises diagnósticas pendentes desde 22/10:
     - **Concordância $Q$ (perplexidade) × $Q_{\text{acc}}$ (acurácia zero-shot média das 5 tarefas):** cálculo de $Q_{\text{acc}} = \text{acc\_media}_q / \text{acc\_media}_{\text{fp}}$, correlação de Spearman/Pearson com $Q$ e taxa de concordância de melhor método por modelo em 4 bits.
     - **Variância de calibração entre sementes (`replicas_awq`):** estatísticas descritivas (média, desvio-padrão $\sigma_{\text{cal}}$, CV%) das sementes `0, 1, 2` do AWQ 4 bits em `Qwen3-0.6B` e `Qwen3-1.7B`.
   - Adicionar testes unitários correspondentes em `experimentos/testes.py`.
3. **Etapa 7A (Fichamento Completo em `literatura/fichas/`):**
   - Criar as 27 fichas de leitura estruturadas segundo `metodologia/ficha-modelo.md` (Análise textual, Análise temática, Análise interpretativa, Problematização, Síntese pessoal, Falseabilidade Popperiana e Citações/metadados verificados) para todos os trabalhos `#01` a `#27` de `literatura/mapeamento.md`, mais um índice `literatura/fichas/README.md`.
4. **Etapa 7B (Tabelas, Figuras e Manuscrito Científico em `artigo/`):**
   - Criar `experimentos/gerar_tabelas_artigo.py` para extrair diretamente dos CSVs e JSONs oficiais todas as tabelas (Markdown e LaTeX) em `artigo/tabelas/` e figuras vetoriais (SVG) em `artigo/figuras/`.
   - Redigir o manuscrito científico completo em `artigo/manuscrito.md` (em português, estrutura ABNT/Severino + Popper conforme `metodologia/estrutura-do-artigo.md` e `metodologia/vertente.md`), onde cada afirmação numérica referencia diretamente as linhas auditáveis dos CSVs.

---

## 3. Critérios de Aceite e Verificação

- **Integridade metodológica (Popper):** Nenhuma regra primária de decisão de `protocolo-v1`, `protocolo-v2` ou `protocolo-v3` é alterada; nenhuma linha válida de `resultados.csv` é sobrescrita sem backup prévio.
- **Testes automatizados:** `uv run python experimentos/testes.py` e `uv run python experimentos/testes_v3.py` passam 100%.
- **Validação de 3 bits (`protocolo-v3` §4.2):** A avaliação do `Qwen3-0.6B-gptq_fp32-3b-g64-s0` produz PPL finita satisfazendo $\text{PPL}_{\text{gptq\_fp32 4b}} (22{,}58) < \text{PPL}_{\text{gptq\_fp32 3b}} < \text{PPL}_{\text{rtn 3b}} (62{,}26)$.
- **Rastreabilidade de dados no artigo:** `uv run python experimentos/gerar_tabelas_artigo.py` gera sem erros todas as tabelas e figuras do artigo a partir de `resultados.csv` e `kquant.csv`.
- **Cobertura bibliográfica:** 27 fichas presentes em `literatura/fichas/` cobrindo 100% das chaves de `literatura/referencias.bib`.
