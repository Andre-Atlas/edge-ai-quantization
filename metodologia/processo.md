# Processo de desenvolvimento em etapas

> Documento central do andamento do estudo: **o que cada etapa constitui**, **o que foi feito** (com data, commit e evidência) e **o que falta**. Atualizado em 25/10/2026. Regra de origem: cada etapa só começa quando a anterior passa na sua validação. **Na prática** as etapas 3, 6 e 7 correram em paralelo à 2 em 22/10 (sessões separadas, sem dependência de dados entre si), e o código da etapa 5 foi escrito antes da execução da etapa 4; a seção 2 registra a ordem real.
>
> Documentos de apoio: `metodologia/protocolo.md` (v1, congelado) e `metodologia/protocolo-v2.md` (vigente: o que mudou e por quê); `metodologia/caderno-de-laboratorio.md` (cronologia passo a passo da etapa 4, raciocínio de cada ação, decisões e incidentes); `experimentos/LOGICA-QUANTIZACAO.md` (lógica do código e validações).

## 1. Visão geral

| Etapa | Constitui | Validação | Estado (25/10/2026) |
|---|---|---|---|
| 0. Ambiente e piloto | `uv` + Python 3.14.9, mlx-lm 0.32.2; pipeline MLX; piloto no Qwen3-0,6B (4 bits) | FP e os 4 métodos rodam de ponta a ponta | ✅ concluída (22/10) |
| 1. Protocolo | `protocolo.md` (tag `protocolo-v1`); `protocolo-v2.md` (tag `protocolo-v2`) | limiares de H1–H4 fixados antes da rodada final; mudanças só por nova versão | ✅ v1 (22/10) · ✅ v2 vigente (25/10) |
| 2. Pipeline robusto | `rodar.py`, `avaliar.py`, `matriz.py`, `testes.py` | `testes.py` passa (determinismo, monotonicidade, 8 bits ≈ FP, tamanho, lm-eval) | ✅ 8/8 testes (22/10); ampliado na etapa 4 |
| 3. llama.cpp (K-quants) | `kquant.py` (Q8_0, Q4_K_M, Q3_K_M, Q2_K com imatrix) | Q8_0 ≤ 1% do F16; PPL monotônica entre os tipos | ✅ validado no 0,6B (22/10); ✅ qualidade dos 5 modelos (25/10); ⏳ velocidade |
| 4. Matriz completa | `resultados/resultados.csv`: qualidade de 89 jobs da v1 (72 após as lacunas da v2) + velocidade isolada | checagens por job; falhas marcadas sem interromper a fila; dados inválidos marcados | 🟡 qualidade 51/72 válidos, 5 inválidos (NaN), 16 a rodar; velocidade não iniciada |
| 5. Análise das hipóteses | `analise.py` → `hipoteses.json`; `analise_h3_nll.py` (secundária) | veredito só pelos critérios do protocolo; IC que cruza o limiar = inconclusiva | 🟡 código pronto e adaptado à v2; `hipoteses.json` final depende da etapa 4 |
| 6. Painel (GitHub Pages) | `site/` gerado por `exportar_site.py`; `.github/workflows/pages.yml` | abre em desktop e celular, sem erro no console, números iguais ao CSV | 🟡 publicado com os dados do **piloto**; atualizar ao fechar a etapa 4 |
| 7. Literatura e redação | rodada 2 de busca, fichas, rascunho pt-BR (ABNT) | cada afirmação aponta para uma linha do CSV; cada citação com DOI/arXiv conferido | 🟡 rodada 2 feita (#14–#27); fichas e redação pendentes |

## 2. Linha do tempo real (linearidade dos experimentos)

| Data | Etapa | Marco | Evidência |
|---|---|---|---|
| 22/10 | — | Estrutura inicial: mapeamento bibliográfico (13 trabalhos) e fundamentação (Popper, Severino) | commit `a33bbc8` |
| 22/10 | 0 | Vertente, análise dos artigos, ambiente `uv`, piloto Qwen3-0,6B; DWQ do piloto | `fbefb28`, `956dd7e`, `resultados/piloto-v0.csv` |
| 22/10 | 1, 2, 3, 6, 7 | Protocolo v1 (tag), pipeline com fases e testes, K-quants (0,6B), painel, rodada 2 da literatura | `a2cf4ad` (tag `protocolo-v1`); sessões "Etapa 3/6/7" |
| 22/10 | 5, 4 | `analise.py` com os critérios da v1; orquestrador da etapa 4; painel com critérios | `b16d534` |
| 22/10 13:34–13:51 | 2 | **Incidente**: o Mac travou duas vezes (3 avaliações em paralelo esgotaram a GPU). Correção: perplexidade uma janela por vez, teto de memória, matriz em série | `40038f7`; `experimentos/README.md` |
| 22/10 | 4 | Agendamento da etapa 4 via `launchd` (00:00, execução única) | `acf5b9d` |
| 23/10 00:01 | 4 | Início da qualidade (89 jobs, um por vez) | `logs/etapa4.log` |
| 23–24/10 | 4 | Paradas e retomadas (pedido do usuário; falta de internet); anomalia de tempo do `0.6B-fp` (8 h 53) | caderno, Fases 1–2 |
| 24/10 | 4 | Lacunas por limitação da ferramenta (GPTQ 3 bits; AWQ no SmolLM3) → mecanismo `excluir` | caderno, Fase 3 |
| 24/10 | 4 | Medição do pico de memória da quantização: estimativa de GPTQ/AWQ estava baixa | caderno, Fase 4 |
| 24/10 | 4 | GPTQ do SmolLM3 com NaN: inspeção dos pesos e teste da Hessiana (causa: bf16) | caderno, Fase 6 |
| 25/10 madrugada | 1, 5 | Rascunho da v2; análise secundária da H3; `analise.py` adaptado | caderno, Fase 7 |
| 25/10 manhã | 4 | Guarda de memória recalibrado; automação (cadeia de jobs, vigia de espera com e-mail) | caderno, Fase 8 |
| 25/10 11h | 4 | GPTQ camada a camada (idêntico ao padrão, 3–4× menos memória) | caderno, Fase 9 |
| 25/10 12–14h | 4 | GPTQ do Llama e do Qwen3-4B também com NaN; causa confirmada nos três modelos | caderno, Fase 10 |
| 25/10 13–16h | 3, 4 | Réplicas de AWQ (ruído ~0,7–0,8%); K-quants dos 5 modelos | caderno, Fase 13 |
| 25/10 15–16h | 4 | DWQ com alvos pré-calculados (não fecha: 20 GB de footprint); AWQ com liberação por bloco (idêntico, 3× menos memória) | caderno, Fases 11–12 |
| 25/10 16:45 | 4 | AWQ a 4 bits do Llama e do Qwen3-4B em execução | `logs/etapa4.log` |
| 25/10 | 1 | `protocolo-v2` vigente (tag); documentação completa; commit e push | commits de 25/10 |

## 3. Requisitos e o que cada etapa constitui

**Etapa 0.** Ambiente reprodutível (`uv.lock`, Python 3.14.9: o 3.15 não tem wheels do PyTorch nem do SentencePiece) e um piloto que prova o pipeline, **sem concluir nada** (uma execução, um modelo).

**Etapa 1.** Definir "qualidade" (`Q = PPL_fp/PPL_q`, mais acurácia zero-shot em 5 tarefas), os critérios de refutação de H1–H4 e a grade experimental, **antes** dos resultados (Popper). O piloto já visto é declarado no próprio protocolo. A v2 registra o que mudou depois da tag sem alterar hipóteses nem limiares.

**Etapa 2.** Fases separadas (*qualidade*, sem medir tempo, e *velocidade*, sempre isolada), retomada, gravação segura com *lock*, semente e commit em cada linha, NLL por janela salva para *bootstrap*.

**Etapa 3.** llama.cpp compilado com Metal e o mesmo texto de calibração; tipos Q8_0, Q4_K_M, Q3_K_M, Q2_K com PPL e KL contra o F16. Só razões contra o próprio F16 são comparadas ao MLX.

**Etapa 4.** 5 modelos × (FP + RTN/GPTQ/AWQ/DWQ × 8/4/3/2 bits) + réplicas de semente do AWQ (4 bits, 0,6B e 1,7B). Qualidade com guarda de memória; velocidade e memória depois, uma por vez, com a máquina ociosa (processos concorrentes dividem GPU e banda de memória e invalidariam a H4).

**Etapa 5.** Bootstrap de 10 000 reamostragens sobre as janelas de PPL; fronteira de Pareto; `hipoteses.json` com corroborada / refutada / inconclusiva e a evidência; regras de contagem da v2 e sensibilidade ao SmolLM3.

**Etapa 6.** Site estático gerado a partir do CSV: contexto, lógica dos testes, gráficos interativos, hipóteses e pontos de melhoria.

**Etapa 7.** Rodada 2025–2026 (DWQ, K-quants, Apple Silicon, KV-cache, MXFP4); fichas de leitura; texto em pt-BR (ABNT) com tabelas e figuras geradas pelos scripts, reutilizáveis na versão em inglês.

## 4. O que foi feito e o que falta, por etapa

### Etapa 0 — ✅
- Feito: ambiente, pipeline, piloto (PPL bf16 20,97 · GPTQ 22,87 · DWQ 23,44 · AWQ 23,99 · RTN 24,33; decodificação ≈ 2× mais rápida, modelo ≈ 3,4× menor).
- Falta: nada.

### Etapa 1 — ✅
- Feito: v1 (tag `protocolo-v1`, 22/10); v2 vigente (tag `protocolo-v2`, 25/10): lacunas da grade, implementações equivalentes de menor memória, falha do GPTQ padrão em 3 de 5 modelos, regras de contagem com dados faltantes, análise secundária pós-hoc da H3, transparência sobre o que já tinha sido visto.
- Falta: nada, salvo se surgir mudança de regra (exigiria v3).

### Etapa 2 — ✅ (ampliada)
- Feito: pipeline e 8/8 testes (22/10); correção do incidente de memória (22/10). Na etapa 4 (25/10): mecanismo `excluir`, guarda de memória recalibrado com medições, medição do pico real de cada quantização (`[pico_quant]`), implementações `gptq_camada.py` e `awq_liberar.py`, coluna `impl`.
- Falta: os testes (`testes.py`) não cobrem as implementações novas; a equivalência delas foi validada à parte (`comparar_gptq.py`).

### Etapa 3 — 🟡
- Feito: `kquant.py`; validação no 0,6B (Q8_0 +0,2% sobre o F16; PPL monotônica); qualidade dos K-quants dos 5 modelos (25/10, `resultados/kquant.csv`).
- Falta: velocidade dos K-quants (`llama-bench`, com a máquina ociosa, dentro da fase de velocidade).

### Etapa 4 — 🟡 (em execução)
- Feito: qualidade de **51 dos 72 jobs** da grade v2 válidos (Qwen3-0,6B e 1,7B completos, com réplicas de AWQ; SmolLM3, Llama e Qwen3-4B com FP e RTN completos); 5 jobs inválidos por NaN (GPTQ do SmolLM3 8/4/2, do Llama 4 e do Qwen3-4B 4); 17 combinações excluídas da grade da v1 (protocolo-v2, seção 3); causa do NaN demonstrada; medições de memória; ferramentas e diagnósticos documentados em `LOGICA-QUANTIZACAO.md`.
- Em execução (25/10, 16:45): AWQ a 4 bits do Llama-3.2-3B e do Qwen3-4B.
- Falta:
  1. AWQ 8/3/2 bits do Llama e do Qwen3-4B (se o de 4 bits for válido).
  2. GPTQ 8 e 2 bits do Llama e do Qwen3-4B (o de 4 bits deu NaN; os de 8 e 2 provavelmente também, pela mesma causa: confirmar ou registrar).
  3. DWQ do SmolLM3 (caso à parte; não cabe com a implementação padrão) e a variante com alvos pré-calculados (protocolo-v2, 7 D), que ainda não fecha (20 GB de footprint no 0,6B).
  4. Linhas antigas de falha do Llama (erro 401, antes da aprovação da Meta): são sobrescritas quando os jobs rodarem.
  5. **Fase de velocidade** (H4), com a máquina ociosa: simular antes (`matriz.py --fase velocidade --simular`), porque uma falha grava `falhou:velocidade` na linha de qualidade válida e a análise a descartaria.
  6. Repetir o tempo do `Qwen3-0.6B-fp` (8 h 53; hipótese de swap; protocolo-v2, 7 C).
  7. Desligar o agendamento `launchd` (`com.edgeai.etapa4`), que continua carregado e tenta disparar todo dia à meia-noite (um marcador impede a repetição).

### Etapa 5 — 🟡
- Feito: `analise.py` com os critérios da v1 (22/10) e as regras da v2 (25/10), sensibilidade ao SmolLM3, testes sintéticos dos ramos; `analise_h3_nll.py` (análise secundária). Vereditos **provisórios** (25/10, ~17h, sem gravar `hipoteses.json`): H1 aguardando (AWQ do Llama e do 4B); H2 inconclusiva (1 de 2 células); H3 inconclusiva sem o SmolLM3 e refutada com ele; H4 aguardando a velocidade.
- Falta (lacunas apontadas em 22/10 e ainda abertas): (a) a acurácia (métrica secundária `Q_acc`) não entra nos vereditos nem como checagem de concordância; (b) as réplicas de AWQ não entram na análise (variância entre sementes calculada só no caderno); (c) o bootstrap mede só a incerteza do texto de avaliação, não a da calibração. Gerar `hipoteses.json` final quando a etapa 4 fechar.

### Etapa 6 — 🟡
- Feito: painel (`site/`), exportador e publicação automática a cada push na `main`; validado localmente.
- Falta: regenerar `site/data/resultados.json` com a matriz completa (decisão de 25/10: manter o piloto até a etapa 4 fechar); conferir que o painel lê o formato real do `hipoteses.json` (`{"hipoteses": {"H1": {"veredito", "resumo", "evidencia"}}}`); testar a renderização com vários modelos; exibir a coluna `impl` e as linhas inválidas como lacunas declaradas.

### Etapa 7 — 🟡
- Feito: rodada 1 (13 trabalhos) e rodada 2 (#14–#27: 12 acadêmicos + 2 documentos técnicos; metadados conferidos; lacuna reformulada em `vertente.md`).
- Falta: fichas (`literatura/fichas/` vazia: nenhum texto completo lido); bases não consultadas diretamente (IEEE Xplore, ACM DL, Springer Link, anais de MLSys/ICLR/NeurIPS/ICML/ACL); *snowballing*; redação em pt-BR (ABNT) com tabelas e figuras geradas dos dados; incluir na discussão a falha do GPTQ padrão e as implementações de menor memória.

## 5. Resultados obtidos até aqui (resumo)

- **Pipeline (etapa 2, Qwen3-0,6B):** RTN repetido dá PPL idêntica (24,3302); PPL monotônica 21,04 (8b) ≤ 24,33 (4b) ≤ 62,26 (3b) ≤ 748 573 (2b); 8 bits a 0,35% do FP.
- **K-quants (etapa 3, Qwen3-0,6B, razão PPL_F16/PPL):** Q8_0 0,998 · Q4_K_M 0,939 · Q3_K_M 0,861 · Q2_K 0,497.
- **Matriz (etapa 4, parcial):** 8 bits sem perda; colapso em 2 bits em todos os métodos; a 3 bits os métodos com calibração superam o RTN com folga; ruído entre sementes do AWQ ~0,7–0,8% de PPL; GPTQ padrão do mlx-lm 0.32.2 falha (NaN) em SmolLM3, Llama-3.2-3B e Qwen3-4B por arredondamento em bf16 da Hessiana.
- **Literatura (etapa 7):** avaliações de sistema em Apple Silicon existem; a lacuna é a comparação lado a lado de algoritmos de PTQ com qualidade e eficiência juntas.
