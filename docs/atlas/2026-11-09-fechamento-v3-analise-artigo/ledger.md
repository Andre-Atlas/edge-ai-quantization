# Ledger de Decisões — 2026-11-09 (`fechamento-v3-analise-artigo`)

## Classificação ATLAS
```
ATLAS · feature · arquitetura+dados+produto · large → pesquisa, descoberta, plano, G1, SDD, revisão, verificação, G2 · lanes: arquitetura
```

## Decisões do Usuário (Descoberta — 2026-11-09)
- **D26 (Workspace):** Mover o repositório completo (163,5 GB incluindo `experimentos/modelos/`, `experimentos/ferramentas/`, `experimentos/logs/` e `.venv/`) de `scratch-workspaces` do Claude para `/Users/andreatlas/Downloads/Edge AI & Quantization`, mantendo symlink de compatibilidade no caminho antigo.
- **D27 (Escopo da Etapa 4b / Protocolo v3):** Focar nos 5 modelos da v2 — concluir imediatamente a avaliação 3 bits já quantizada do `Qwen3-0.6B` e executar apenas o Bloco 1 da `matriz-v3.json` (`gptq_fp32` em 4 e 3 bits nos 5 modelos da v2) para comprovar a correção da Hessiana em `fp32` sem comprometer o prazo de 18/12/2026.
- **D28 (Frentes Prioritárias):**
  1. Fechar pendências analíticas da Etapa 4/5 (avaliar `Qwen3-0.6B gptq_fp32 3b`, adicionar concordância $Q \times Q_{\text{acc}}$ zero-shot e variância das réplicas AWQ em `analise.py`).
  2. Elaborar as 27 fichas de leitura em `literatura/fichas/`.
  3. Iniciar a Etapa 7 — Redação do manuscrito científico (ABNT pt-BR) em `artigo/` e scripts de geração de tabelas/figuras.
- **D29 (Regra Global Antigravity):** Padronizar globalmente em `~/.gemini/config/AGENTS.md` e `~/.gemini/config/skills/atlas-maestro/references/antigravity.md` o uso da ferramenta `ask_question` com *single/multi-choice checkboxes* e prefixo `(Recommended)` na primeira opção.

## Rulings Intermediários
- `Ruling: Manter um symlink no caminho antigo do scratch-workspaces apontando para /Users/andreatlas/Downloads/Edge AI & Quantization — evita quebra de qualquer script, plist ou referência absoluta pré-existente — custo zero se estiver errado (é apenas um link simbólico).`
- `Ruling: Filtrar analise.py (protocolo-v2) explicitamente por MODELOS da v2 e metodos {"fp", "rtn", "gptq", "awq", "dwq"} em carregar() — evita que linhas de gptq_fp32 da extensão v3 alterem n_configuracoes (59 -> 60) ou a fronteira de Pareto da v2 em hipoteses.json — custo zero e preserva a separação estrita v2 vs v3 exigida pelo protocolo-v3.`

## Fase 5 & 6 — Revisão e Verificação (2026-11-09)
- **Revisão Arquitetural (`architect`):** Aprovado integralmente — confirmou o isolamento entre `protocolo-v2` e `protocolo-v3`, o pipeline reprodutível de artefatos (`gerar_tabelas_artigo.py` -> `.md`/`.tex`/`.svg` -> `manuscrito.md`) e a conformidade epistemológica (Popper/Severino) das 27 fichas em `literatura/fichas/`.
- **Revisão Python (`python-reviewer`):** Apontou melhorias de abertura de arquivo com `with open(..., encoding="utf-8")`, proteção contra variância nula em `_calcular_corr`, proteção contra divisão por zero via `_safe_div`, tipagem estática PEP 484 e expansão de testes de borda em `testes_v3.py`. Todas as correções foram aplicadas e validadas (`29/29` asserções passando em `testes_v3.py`).
