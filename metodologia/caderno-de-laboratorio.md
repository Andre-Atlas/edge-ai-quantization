# Caderno de laboratório — etapa 4 (execução da matriz de quantização)

> Registro cronológico, em ordem, de **cada passo** da execução da etapa 4 e das decisões tomadas, com motivo, evidência e resultado. Complementa `experimentos/LOGICA-QUANTIZACAO.md` (lógica do código e validações) e `metodologia/protocolo-v2.md` (regras de decisão; vigente desde 25/10/2026). O que cada etapa do projeto constitui, o que foi feito e o que falta está em `metodologia/processo.md`. Cobre de 22/10/2026 (contexto) a 25/10/2026, ~18h.
>
> **Convenções.** *Usuário* = decisão do pesquisador responsável; *Assistente* = Claude Code (execução, análise, código). Horários entre parênteses vêm de `experimentos/logs/etapa4.log` ou dos horários de criação dos arquivos; onde só há a data, o horário não foi registrado. Memórias em GB decimais (§0 de `LOGICA-QUANTIZACAO.md`). "Fonte" aponta o arquivo que comprova o fato. Nada aqui foi reescrito depois de ver o resultado: os erros e as correções de curso estão na seção 4.

## 1. Cronologia

### Fase 0 — Contexto herdado (22/10)
- Protocolo v1 congelado na tag `protocolo-v1`: hipóteses H1–H4, limiares, grade (5 modelos × 4 métodos × bits 8/4/3/2, grupo 64). Piloto no Qwen3-0,6B (`resultados/piloto-v0.csv`, visto antes dos limiares, declarado no protocolo).
- **Incidente de 22/10 (13:34 e 13:51):** o Mac travou duas vezes com 3 avaliações em paralelo (picos de 12 GB por processo de perplexidade, estimativa subestimada, usuário em uso). Correções já aplicadas (README de `experimentos/`): perplexidade uma janela por vez, teto de memória por processo, estimativa conservadora.
- Etapa 4 agendada: `executar_etapa4.sh 1` (qualidade, 1 job por vez → K-quants → velocidade → análise). Início **23/10 00:01:58**, memória livre 9,6 GB. Fonte: `logs/etapa4.log`.

### Fase 1 — Análise de energia e primeira parada (23/10)
1. **Pedido do usuário:** analisar o sistema por processos que gastam bateria. Mac na tomada, 55% e carregando.
2. **Achados (Assistente):** (a) a etapa 4 (ligada havia ~11h30) avaliando o `Qwen3-0.6B gptq-4b` (58–71% de CPU, ~1,6 GB, GPU); (b) um `caffeinate -i -s -t 35508` esquecido (PID 2227, ~21 h) sem tarefa associada; (c) uso normal (WindowServer, `kernel_task`). Opções A–D apresentadas; **usuário escolheu A** (só o `caffeinate` esquecido).
3. O `kill` foi **bloqueado** pelo classificador de permissões (ação sobre processo em execução). O usuário executou `kill 2227` no Terminal; o Assistente confirmou o encerramento.
4. **Estado da etapa 4 (leitura dos logs):** 6 de 89 avaliações; tempos por job; **`Qwen3-0.6B-fp` levou 31 962 s (8 h 53)** contra ~25 min do `1.7B-fp` (hipótese de swap, **não confirmada**); **GPTQ 3 bits falha** (`AssertionError: Unsupported bits 3`, limitação do mlx-lm).
5. **Parada pedida pelo usuário.** O Assistente enviou SIGTERM ao grupo de processos da etapa 4 e conferiu que nada ficou rodando. Resultados preservados (a execução é retomável).
6. **Retomada (23/10 17:53:34)**, comando em segundo plano (`nohup`, `caffeinate -i -s`), independente do app.
7. **Anomalia do fp:** `repetir_fp_0.6B.sh` criado (23/10 20:01) para repetir a avaliação no fim, gravando em CSV separado. **Foi cancelado** quando a etapa foi parada na Fase 2 e **não foi reagendado** (pendência).

### Fase 2 — Sem internet, paradas e retomadas (24/10)
- **24/10 09:19:** estado: SmolLM3 GPTQ rodando (~1h45 por job), bateria descarregando (87%), swap 10,5 de 11,3 GB.
- **09:22:06:** usuário avisa que ficará sem internet → o Assistente parou a fila e o script que repetiria o fp (nessa ordem, para o script de espera não disparar). `resultados.csv` íntegro (39 linhas).
- **09:57:27:** retomada pelo usuário (Mac na tomada).
- Esclarecimento ao usuário: fechar a tela suspende o Mac mesmo com `caffeinate`; recomendado tampa aberta, na tomada.

### Fase 3 — Falhas por limitação da ferramenta; mecanismo `excluir` (24/10, 19h–20h)
1. **19:41 (leitura dos logs):** a fila ficou parada desde 16:07 esperando memória (214 avisos: `SmolLM3-3B dwq` exige 18,4 GB livres, havia 16,1). Dois achados: **AWQ não suporta o SmolLM3** (`NotImplementedError … smollm3 … NYI`, 4 jobs em 4 s); **GPTQ 3 bits** (já visto).
2. **Usuário:** "ajuste a memória e feche os arquivos desnecessários; aceito as sugestões acerca do artigo". O Assistente tentou fechar o Ollama (ação cancelada pelo usuário no app; **não foi forçado**); o usuário o fechou depois. A memória livre não chegou ao exigido.
3. **Mecanismo `excluir` criado** em `matriz.json`/`matriz.py` (M1): GPTQ 3 bits (todos) e AWQ no SmolLM3. Simulação: 40 jobs pendentes. Fila reiniciada às **19:46:42** (nenhum job interrompido). Lacunas registradas no protocolo (notas em `protocolo.md`, hoje consolidadas no v2).
4. **Pergunta do usuário:** "a memória exigida é especulativa?" Resposta com dados: estrutura baseada no incidente de 22/10, mas multiplicadores **não calibrados**; picos medidos só da etapa de perplexidade.
5. **Usuário: opção A** — rodar o que cabe (fp, rtn, gptq, awq) e deixar DWQ por último. Fila reiniciada às **19:51:25** só com `--metodos rtn,gptq,awq`.
6. **Llama-3.2-3B falhou em 4 s por erro 401:** modelo restrito (*gated*) no Hugging Face (ver Fase 5).

### Fase 4 — Medição do pico de memória da quantização (24/10, 19:54–20:20)
1. **Usuário:** medir o pico da quantização no Qwen3-1,7B. Fila **pausada às 19:54:45**.
2. `medir_pico_quant.py` criado (LOGICA §3.5): uma quantização por processo, pico do MLX, amostragem de memória do sistema, trava de segurança.
3. **Resultados (Qwen3-1,7B, 4 bits):** GPTQ **14,66 GB** (previsto 10,1); AWQ **24,20 GB** (previsto 10,1; swap +7,6 GB); DWQ **abortado** pela trava (pico > 9,56 GB e subindo). **A estimativa antiga estava baixa**, e a sugestão do Assistente de que o limite fosse "conservador demais" estava **errada para GPTQ/AWQ** (corrigido na hora; ver §4).
4. **Usuário: opção A de matriz** — limitar o Qwen3-4B a fp e rtn (GPTQ/AWQ/DWQ excluídos por memória). Registrado em `matriz.json` e no protocolo (notas).
5. Retomada **20:20:46** só com fp e rtn.

### Fase 5 — Acesso ao Llama-3.2-3B e token do Hugging Face (24/10, ~20h)
- **Usuário** fornece token. O Assistente **não** leu nem pediu o token no chat: instruiu `hf auth login` no terminal do usuário (token salvo em `~/.cache/huggingface/token`).
- Teste de leitura: 401 → 403 (autenticado, mas **licença da Meta não aceita**). O usuário solicitou acesso (status **PENDING** na página do Hugging Face); a aprovação saiu em cerca de 10 minutos (a fila já avaliou o Llama a partir das 20:20).
- **Usuário:** o primeiro token tinha acesso irrestrito → instrução registrada: **verificar 3 vezes antes de agir** (memória do assistente). Usuário criou um segundo token só de leitura (`Article_DEV_READ`, *fine-grained*) e fez a troca; o Assistente confirmou o login sem ver o token.

### Fase 6 — Inspeções leves e teste da Hessiana no SmolLM3 (24/10 noite)
1. **Priorização pedida pelo usuário** (organizar os testes por prioridade e os casos a reanalisar). Apresentada em tabela (P0–P5 e 6 casos).
2. **Inspeções leves** (sem competir com a fila): (a) pesos do GPTQ do SmolLM3: **NaN/Inf em 8 tensores** (escalas e *biases* de `o_proj` camadas 0, 5, 34 e `down_proj` 10), **idênticos para 8, 4 e 2 bits** → o NaN nasce na calibração; 31 de 32 grupos NaN por tensor; (b) perplexidade por janela: NaN em todas as 141; (c) RTN 3 bits do SmolLM3 (Q = 0,09): **real**, sem NaN (4,61 nats contra 2,21 do fp); (d) colapso de 2 bits: **real**, finito (13,4 nats nos Qwen3 e 25,1 no SmolLM3, acima do chute uniforme 11,8).
3. **Usuário:** marcar o GPTQ do SmolLM3 como inválido e testar a hipótese da Hessiana. Linhas marcadas `falhou:invalido-nan` (com backup: `backups/resultados.csv.antes-de-invalidar-gptq-smollm3`).
4. `testar_hessiana.py` criado (**24/10 20:36**): 35 camadas do SmolLM3, camada a camada. **Resultado:** só as mesmas 4 matrizes falham com a Hessiana em bf16; em float32 **nenhuma**.

### Fase 7 — Protocolo v2 (rascunho) e regras (25/10 madrugada–manhã)
- **04:07** — Usuário: "C agora, A depois, B por último se os achados de A forem consideráveis". C = registrar a falha como resultado (notas em `protocolo.md`, **alteração após a tag**, a ser consolidada na v2).
- **04:56** — Sugestão (Assistente) sobre a saturação de Q em 2 bits na H3; análise secundária pós-hoc `analise_h3_nll.py` (LOGICA §3.7). O **critério original falha por teto** em 3 dos 4 modelos (**não** no Llama); a escala logarítmica mostra superlinearidade nos 4.
- **Rascunho do protocolo-v2** criado (`metodologia/protocolo-v2.md`), com transparência (o que já tinha sido visto), lacunas, regras de contagem, análise secundária e análises A/B. Aprovado pelo usuário.
- **05:05** — `analise.py` adaptado à v2 (LOGICA §2.4); testes sintéticos de ramos; um erro meu na contagem do RTN em H1 foi encontrado e corrigido nos testes.
- **Usuário:** SmolLM3 como caso à parte; limiares absolutos mantidos; AWQ/DWQ do Llama excluídos (o AWQ do Llama foi **reabilitado depois**, Fase 12).
- **05:11** — `agendar_llama_gptq.sh`: GPTQ do Llama a 4 bits, sozinho, depois da fila de RTN.

### Fase 8 — Guarda de memória e automação (25/10, 09h–12h)
- **Usuário:** recalibrar o guarda de memória. Primeiro **fp e rtn** (22 linhas medidas → regra `pico = tamanho + 2,7`, conversão ≤ tamanho bf16), depois **GPTQ, AWQ e DWQ** (LOGICA §4). Confirmado: a memória do processo (*footprint*) é 0,6–2,3 GB maior que o pico do MLX → rodar também `time -l`. **Unidades:** RAM = 24 GiB = 25,77 GB decimais (esclarecido a pedido do usuário).
- **09:35** — `vigiar_espera.sh`: alerta de espera > 1 h (com e-mail ao usuário); **09:37** — `continuar_lista.sh` (réplicas de AWQ e K-quants).
- Nova **regra de memória do `rodar.py`**: linha `[pico_quant]` em todo job (R1). O GPTQ do Llama (padrão) forneceu o primeiro ponto fora da amostra: **26,06 GB** de footprint contra 40,2 previstos (a regra superestima em modelos grandes).

### Fase 9 — GPTQ camada a camada (25/10, 11:13–11:50)
- **Usuário:** "escreva o GPTQ camada a camada e valide no 0,6B". `gptq_camada.py` (**11:13**) e `comparar_gptq.py` (**11:16**) criados (LOGICA §3.1).
- **Validação:** 704/704 tensores idênticos e NLL com diferença 0 no 0,6B e no 1,7B; pico 3–4× menor. **Modo float32 não reproduz o padrão** (razão Q 1,0143; IC 1,0127–1,0159) → análise A vira **diagnóstico**, e B perde o efeito.
- **Usuário:** rodar o GPTQ do Qwen3-4B (4 bits) por esse caminho. `matriz.json`: exclusão do GPTQ do 4B **removida**; `matriz.py` passa a escolher a implementação (`usa_camada`). Agendado (**11:48**), depois do Llama.

### Fase 10 — GPTQ dos modelos grandes: NaN (25/10, 12:07–14:13)
- **Llama-3.2-3B GPTQ 4 bits** (padrão, 2 522 s): **PPL = NaN**, acurácia 0,3438. **Qwen3-4B GPTQ 4 bits** (camada a camada, 930 s): **PPL = NaN**. Marcados `falhou:invalido-nan` (backup `…antes-de-invalidar-gptq-llama-qwen4b`).
- Consequência (v2, seção 4.1): o GPTQ padrão falha em **3 de 5 modelos**; **a H1 só tem 2 modelos avaliáveis** (Qwen3-0,6B e 1,7B) e só pode sair inconclusiva, a menos que AWQ ou DWQ voltem a caber.
- `testar_hessiana.py` estendido (CPU, 4 matrizes por camada) e rodado no Llama e no Qwen3-4B (**14:19–16:17**): o mesmo mecanismo (§ LOGICA 3.2). Llama: 1 matriz; Qwen3-4B: 7 matrizes; em float32, nenhuma.

### Fase 11 — DWQ: alvos do professor (25/10, 12:00–16:30)
1. Leitura do `dwq.py`: o `--target-dir` dispensa o professor no treino, mas guarda só os 1024 maiores logits (**perda truncada → resultado não idêntico ao padrão**) e o cálculo dos alvos tem pico ∝ lote × tokens × vocabulário.
2. **Teste 1 (11:57) e 2 (12:00/13:24):** abortados em 12 e 16 s pela trava (a 1ª com outros jobs; a 2ª, sozinha: **footprint 19,8 GB só na fase de alvos**, num modelo de 0,6B). Erros intermediários de execução (§4).
3. **`dwq_sublote.py`** (**15:46**): alvos em sub-lotes de 1; equivalência verificada na CPU (índices 100%, diferença 0).
4. **Diagnóstico real (16:18):** alvos concluídos (6,6 GB em disco); treino até ~400/512 passos, pico MLX **5,89 GB**, mas footprint **20,05 GB** → abortado pela trava antes de salvar. **Sem comparação de qualidade.**

### Fase 12 — AWQ: causa da memória e correção (25/10, 15:46–16:48)
1. **`perfil_awq.py`** (**15:46**) e perfil no Qwen3-0,6B (16:27): a memória ativa **cresce ~1,4 GB por bloco** (= tamanho do `input_feat` do bloco); leitura do `awq.py`: nunca é apagado.
2. **`awq_liberar.py`** (**16:34**): `gc.collect()` antes de `mx.clear_cache()` + limite do cache do MLX (1 GB). **Validação:** 704/704 tensores idênticos e NLL com diferença 0 no 0,6B (pico 13,42 → 4,17 GB; footprint 5,43 GB) e no 1,7B (pico 24,2 → 8,44 GB; footprint 9,30 GB).
3. **Integração (16:44):** `--awq-liberar` em `rodar.py`; `matriz.py` passa a usá-lo sempre para AWQ, com estimativa `4,0 + 1,4 × fp_gb`; `matriz.json`: exclusão do AWQ do Llama **removida** (a do SmolLM3, por limitação da ferramenta, permanece); `agendar_awq_grandes.sh` (**16:45**).
4. **Usuário** pediu verificar outras modificações que melhorem o resultado. Avaliadas: (i) compartilhar as cópias idênticas de `input_feat` de q/k/v e gate/up (economia adicional dentro do bloco; **não implementado**, exige copiar o interior do `awq_quantize`; só se o 4B não couber); (ii) tempo: igual; (iii) qualidade: **não alterada** de propósito.
5. **Em andamento (16:45):** AWQ 4 bits do Llama-3.2-3B e depois do Qwen3-4B (estimativas de ~13 e ~15 GB, a validar com `[pico_quant]`).

### Fase 13 — Réplicas de AWQ e K-quants (25/10, 13:25–16:15)
- Réplicas de AWQ a 4 bits (sementes 1 e 2): Qwen3-0,6B PPL 23,62 / 23,96 (semente 0: 23,72; desvio 0,8%); Qwen3-1,7B 17,12 / 17,34 (semente 0: 17,23; desvio 0,7%).
- K-quants (llama.cpp) concluídos para os cinco modelos (25 linhas em `kquant.csv`). **Fase de velocidade (H4) não foi iniciada** (exige a máquina ociosa).

### Fase 14 — Protocolo v2 vigente, documentação do projeto e publicação (25/10, ~17h–18h)
1. **Usuário:** renomear esta conversa para "Etapa 4" e indexá-la com as outras etapas; ler o histórico de todo o projeto; reunir a documentação; commit e push, com foco na linearidade dos experimentos e no que cada etapa constitui, fez e falta.
2. **Barra lateral:** conversa renomeada para "Etapa 4: execução da matriz de quantização". Não existe operação para tornar uma conversa "filha" de outra depois de criada (as etapas 3, 6 e 7 são filhas porque foram abertas pela sessão principal); **usuário escolheu** criar o grupo "Edge AI & Quantization article" com a sessão principal, as etapas 3, 6, 7 e esta.
3. **Histórico lido:** os 7 commits (22/10), a sessão principal (incluindo três lacunas do `analise.py` apontadas em 22/10 e ainda abertas: acurácia fora dos vereditos, réplicas de AWQ ignoradas, bootstrap sem a variação da calibração), as sessões das etapas 3, 6 e 7 (pendências de cada uma) e toda a documentação.
4. **Decisões do usuário (com recomendação do Assistente):** push direto na `main` (padrão do repositório; dispara a publicação do painel); **manter o painel com os dados do piloto** até a etapa 4 fechar; **restaurar `protocolo.md` à v1 e criar a tag `protocolo-v2`**; agrupar as sessões.
5. **Protocolo v2 vigente:** antes de restaurar a v1, cada nota acrescentada a ela foi conferida na v2. A v2 foi atualizada com fatos posteriores ao rascunho (AWQ com liberação por bloco, seção 3.2; causa do NaN confirmada nos três modelos; `analise.py` já adaptado) e com a **transparência sobre os vereditos provisórios vistos antes da tag** (seção 2). `protocolo.md` restaurado byte a byte ao da tag.
6. **Dados atualizados antes do commit:** `analise_h3_nll.py` rodado de novo com o RTN do Qwen3-4B completo (S > 0 nos cinco modelos); vereditos provisórios: ver seção 5.
7. **Documentos do projeto:** `metodologia/processo.md` reescrito (visão geral, linha do tempo real com commits e fases, o que cada etapa constitui, feito e falta por etapa); `README.md` da raiz atualizado (mapa da documentação, estado das etapas, estrutura); este caderno e `LOGICA-QUANTIZACAO.md` atualizados.
8. **Commits** separados por tema, em ordem lógica (pipeline → ferramentas → protocolo e análise → dados → documentação), com a tag `protocolo-v2`; push para `origin/main`. O `resultados.csv` é uma fotografia do momento do commit: o AWQ do Llama estava em execução e grava no mesmo arquivo.

## 1A. Raciocínio por trás de cada ação

> Para cada ação: **por que** foi feita, e qual **alternativa** foi descartada e por quê. O raciocínio é o que foi apresentado ao usuário **no momento** da ação (conversa da sessão), não uma justificativa reconstruída depois; onde houve reconstrução, está marcado *(reconstruído)*. Princípios que se repetem: (P1) **não mudar o método depois de ver resultados** (pré-registro, Popper); (P2) **segurança da máquina antes de completar uma medição** (incidente de 22/10); (P3) **preservar o dado bruto** e excluir da análise só por regra explícita; (P4) **medir antes de decidir** quando a decisão depende de um número; (P5) **decisões que mudam o estudo são do usuário**; o Assistente propõe, com alternativas.

### Fase 1 — energia e primeira parada
| Ação | Por quê | Alternativa descartada |
|---|---|---|
| Recomendar matar só o `caffeinate` esquecido | Não tinha tarefa associada: custo zero de perda. Parar o experimento perderia a avaliação em curso e é decisão do usuário (P5) | Parar o experimento: perda de trabalho sem necessidade |
| Não contornar o `kill` bloqueado | O bloqueio veio do sistema de permissões; contorná-lo violaria a intenção do bloqueio | Matar por outro caminho: o usuário executou o comando ele mesmo |
| Encerrar a etapa 4 com SIGTERM ao **grupo** de processos e conferir que nada sobrou | A etapa é uma árvore (zsh → caffeinate; `matriz.py` → `rodar.py` → `mlx_lm`/`avaliar.py`); matar só o pai deixaria filhos órfãos usando a GPU. SIGTERM (e não SIGKILL) permite encerramento limpo | Matar PIDs um a um: arrisca esquecer filhos |
| Retomar com `nohup` + `caffeinate -i -s` + `disown` | O processo sobrevive ao fechamento do app e o Mac não dorme por inatividade. Fechar a tampa ainda suspende, daí a orientação ao usuário | Rodar ligado à sessão: morreria com o app |
| Repetir o `0.6B-fp` **no fim** e em **CSV separado** | A qualidade é determinística: só o tempo era anômalo. CSV separado não sobrescreve o resultado oficial; rodar no fim evita contaminar o tempo com outros jobs | Repetir na hora, no CSV principal: mistura medição de tempo com fila e reescreve dado (P3) |
| Hipótese de swap para as 8h53 | O `1.7B-fp` (3× maior) levou 25 min, e o swap estava alto no início. **Não confirmada** | — |

### Fase 2 — sem internet
| Ação | Por quê | Alternativa descartada |
|---|---|---|
| Parar o script de espera **antes** da fila | Se a fila morresse primeiro, o script entenderia "etapa terminou" e iniciaria a repetição do fp sem querer | Parar a fila primeiro |
| Garantir que a retomada não dependia de internet | Modelos e dados já estavam em cache; só um modelo novo baixaria algo | — |

### Fase 3 — limitações da ferramenta e `excluir`
| Ação | Por quê | Alternativa descartada |
|---|---|---|
| Criar o mecanismo `excluir` | O GPTQ 3 bits do SmolLM3 levou 43 min para falhar, e a cada retomada a matriz tentaria de novo. A exclusão explícita documenta a lacuna com o motivo | Deixar falhar: tempo perdido e ruído no CSV |
| **Não** substituir por outro método ou configuração | Mudaria o desenho depois de ver resultados (P1) | Trocar por K-quant ou por outro número de bits |
| Não forçar o fechamento do Ollama | A ação foi cancelada pelo usuário na confirmação | Forçar o encerramento |
| Não baixar o limiar de memória naquele momento | O limiar existia por causa do travamento de 22/10; mudar sem dados repetiria o erro (P2, P4). Propor **medir** | Reduzir na intuição |
| Opção A (rodar primeiro o que cabe) | Um job que nunca cabia (DWQ do SmolLM3) bloqueava toda a fila atrás dele | Esperar memória |

### Fase 4 — medição de memória
| Ação | Por quê | Alternativa descartada |
|---|---|---|
| Medir no **Qwen3-1,7B** | Tamanho intermediário: cabe com folga para medir e permite extrapolar. O 4B arriscaria travar | Medir direto no 4B |
| **Pausar** a fila durante a medição | Memória compartilhada: concorrência contaminaria o pico e somaria demanda acima da RAM (P2) | Medir junto da fila |
| Medir em processo próprio, com a mesma linha de comando do pipeline | Mesmo código que o estudo usa (validade). O pico do MLX só é legível dentro do processo | Estimar pelo `ps`: o `ps` não vê a memória da GPU |
| Trava de 1,5 GB livres (3 GB quando há outros jobs) | Em 22/10 o travamento veio de falta de memória de GPU; abortar cedo custa só a medição (P2) | Sem trava |
| Limitar o Qwen3-4B a fp/rtn (decisão do usuário) | A extrapolação punha GPTQ/AWQ acima da RAM; foi a opção mais segura e honesta apresentada. Reabrir se surgisse método validado (o que ocorreu) | Reduzir a calibração: mudaria o método (P1) |

### Fase 5 — token do Hugging Face
| Ação | Por quê | Alternativa descartada |
|---|---|---|
| Não pedir nem ler o token; login pelo próprio usuário | Credencial do usuário: menor exposição possível | Colar o token no chat |
| Testar com um download pequeno (`config.json`) | Verifica o acesso sem baixar gigabytes | Rodar o job e ver se falha |
| Sugerir token só de leitura | Menor privilégio necessário para o estudo | Manter o token irrestrito |

### Fase 6 — inspeções e Hessiana do SmolLM3
| Ação | Por quê | Alternativa descartada |
|---|---|---|
| Inspeções **leves** primeiro | Custo baixo de memória, rodam junto da fila, e dizem se vale gastar ~1h45 refazendo | Refazer o GPTQ direto |
| Comparar os tensores com NaN entre 8, 4 e 2 bits | Se o NaN viesse da quantização em si, dependeria dos bits. Idênticos indicam que vem antes, da calibração | — |
| **Marcar como inválido** (não apagar), com backup | Preserva o dado bruto para auditoria e o exclui da análise por regra explícita (P3) | Apagar as linhas |
| Teste da Hessiana camada a camada, comparando bf16 com fp32 e olhando o menor autovalor | Reproduz a coleta com ~5 GB em vez de ~22 GB. Isola a **precisão** como única variável. O autovalor separa "matriz indefinida" de "mal condicionada" | Refazer o GPTQ inteiro em fp32 |

### Fase 7 — protocolo v2 e regras
| Ação | Por quê | Alternativa descartada |
|---|---|---|
| Relatar a falha do GPTQ como resultado (C) | A falha é um achado legítimo; consertar primeiro atrasaria e mudaria o método (P1) | Corrigir antes de relatar |
| H3: manter o critério original como veredito e acrescentar análise em escala log **declarada pós-hoc** | Trocar a métrica depois de ver o resultado, para salvar a hipótese, é o ajuste *ad hoc* que Popper critica (P1) | Trocar o critério |
| Escrever uma **v2** em vez de editar a v1 | A própria v1 exige nova versão para qualquer mudança | Editar a v1 (feito por engano com notas; ver E11) |
| Manter limiares absolutos (usuário; recomendação do Assistente) | Mudar limiares depois de ver dados é o que o pré-registro quer evitar. O custo aceito é "inconclusiva" | Limiares proporcionais |
| SmolLM3 à parte, com **sensibilidade** calculada (usuário) | Sem método válido no modelo. A exclusão muda a 1ª metade da H3, então as duas versões precisam ser reportadas | Excluir sem reportar o efeito |
| Testes sintéticos do `analise.py` | A lógica de veredito tem muitos ramos que os dados reais não exercitam. Um erro foi achado assim | Testar só com dados reais |
| GPTQ do Llama **sozinho** | Estimativa de ~22+ GB: concorrência poderia travar (P2) | Rodar junto da fila |

### Fase 8 — guarda de memória e automação
| Ação | Por quê | Alternativa descartada |
|---|---|---|
| Recalibrar fp/rtn com 22–24 picos medidos | A fila perdeu ~4,5 h por exigir mais memória do que o maior job usou (P4) | Manter a regra antiga |
| Usar também o **footprint** do processo | O MLX não conta o cache nem a memória fora dele; o sistema trava pelo total | Usar só o pico do MLX |
| Teto de partida = metade da RAM | Jobs maiores que a RAM já rodaram sozinhos com swap sem travar; exigir o pico inteiro livre os bloquearia para sempre. O teto equivale ao limiar que já tinha funcionado *(reconstruído em parte: o histórico de limiares foi conferido depois)* | Exigir o pico inteiro |
| Linha `[pico_quant]` em todo job | Validação contínua fora da amostra, sem custo extra | Medir só em testes |
| Vigia de espera > 1 h, com e-mail | Evitar novas horas paradas sem ninguém ver; o usuário pediu avisos | Conferir manualmente |
| Exigir que a regra nunca subestime nenhuma medição | Condição de segurança: errar para cima custa espera, errar para baixo pode travar | Ajuste por média |

### Fase 9 — GPTQ camada a camada
| Ação | Por quê | Alternativa descartada |
|---|---|---|
| Critério de equivalência = **tensores bit a bit + NLL por janela** | Mais forte que comparar a perplexidade: se os tensores são idênticos, qualquer métrica derivada também é | Comparar só a PPL |
| Validar em **dois** tamanhos | Excluir coincidência num único modelo | Validar só no 0,6B |
| Classificar o fp32 como diagnóstico | A regra foi escrita **antes** do teste (v2, seção 7) e foi aplicada como escrita | Reinterpretar a regra depois do resultado |
| Rodar o 4B pelo camada a camada | Equivalente bit a bit: é o mesmo método, não um desvio | Manter o 4B excluído |
| A matriz escolhe a implementação sozinha (`usa_camada`) | Impedir que alguém rode a versão padrão, estimada em 57 GB | Depender de lembrar a opção |

### Fase 10 — NaN nos modelos grandes
| Ação | Por quê | Alternativa descartada |
|---|---|---|
| Marcar Llama e Qwen3-4B como inválidos | Mesma regra aprovada para o SmolLM3; sem isso o `analise.py` leria NaN como "abaixo de 0,95" | Deixar no CSV como válidos |
| Teste da Hessiana na **CPU** e com prioridade baixa | O AWQ do 1,7B ocupava a GPU; disputar GPU foi a causa do travamento de 22/10 (P2) | Rodar na GPU ou esperar ~2 h |
| Rodar **todas** as camadas | O padrão de quais matrizes falham só aparece completo | Amostrar camadas |

### Fase 11 — DWQ
| Ação | Por quê | Alternativa descartada |
|---|---|---|
| Ler o `dwq.py` antes de propor | Saber o custo real: os alvos guardam só 1024 logits, então o resultado não é idêntico | Propor pelo nome da opção |
| Testar no 0,6B | Barato; responde "cabe ou não" antes de gastar no 4B | Testar direto no 4B |
| Alvos em sub-lotes | As linhas são independentes, então os alvos são os mesmos com ~¼ do pico da fase de alvos | Reduzir o lote do treino: muda a dinâmica (P1) |
| Verificar a equivalência na CPU | A GPU estava ocupada, e a CPU dá um resultado determinístico para checar a lógica | Verificar só no treino completo |
| Aceitar o aborto pela trava, sem afrouxá-la | A segurança tem precedência sobre completar a medição (P2) | Baixar a trava |

### Fase 12 — AWQ
| Ação | Por quê | Alternativa descartada |
|---|---|---|
| Perfil de memória **antes** de corrigir | Localizar o problema em vez de chutar (P4) | Reduzir amostras: mudaria o método (P1) |
| Correção mínima (`gc.collect` + limite de cache) | Não toca nenhum cálculo; a validação bit a bit é exigida | Reescrever o AWQ em pedaços |
| Não compartilhar as cópias de q/k/v | Custo e risco maiores que o necessário; só se o 4B não couber | Implementar já |
| AWQ sempre liberado na matriz | Equivalência provada em dois modelos; reduz o risco de travar | Usar só quando não coubesse |
| Remover a exclusão do AWQ do Llama e manter a do SmolLM3 | O motivo da exclusão do Llama (memória) deixou de valer; o do SmolLM3 (a ferramenta não suporta) continua | Remover as duas |

### Documentação (Fase 13)
| Ação | Por quê | Alternativa descartada |
|---|---|---|
| Usar os horários de criação dos arquivos e as linhas do log como fonte de tempo | Fontes objetivas, independentes da memória da conversa | Reconstruir de memória |
| Copiar os backups para o projeto, com SHA-256 | A pasta temporária some com a sessão; o checksum prova a integridade | Deixar onde estavam |
| Registrar os erros do Assistente (seção 4) | Um registro que só mostra acertos não serve para auditoria | Omitir |

### Fase 14 — documentação e publicação
| Ação | Por quê | Alternativa descartada |
|---|---|---|
| Recomendar push direto na `main` | Os 7 commits anteriores foram direto na `main`; uma branch deixaria o painel e o histórico principal desatualizados | Branch + PR (oferecida ao usuário) |
| Recomendar manter o painel com o piloto | A matriz está incompleta e os vereditos são provisórios; publicar dados parciais num site público pode ser lido como resultado final | Publicar já, com aviso |
| Conferir cada nota da v1 na v2 **antes** de restaurar a v1 | Restaurar descarta texto; só é seguro se o conteúdo existir em outro lugar | Restaurar direto |
| Registrar na v2 os vereditos provisórios vistos antes da tag | Congelar regras depois de ver um veredito sem dizer isso esconderia um grau de liberdade do pesquisador (P1) | Omitir |
| Commits separados por tema e em ordem lógica | Cada commit conta uma parte da história (o que mudou e por quê) e pode ser revisto ou revertido sozinho | Um commit único |
| Grupo na barra lateral em vez de "filha" | Não existe operação para vincular uma conversa existente como filha | — |

## 2. Registro de decisões

| # | Data | Decisão | Quem | Alternativas consideradas | Efeito |
|---|---|---|---|---|---|
| D1 | 23/10 | Matar só o `caffeinate` esquecido | Usuário | parar o experimento; ambos; nada | Mac volta a dormir ao fim da etapa 4 |
| D2 | 23–24/10 | Parar e retomar a etapa 4 (parada às 09:22 de 24/10 por falta de internet) | Usuário | — | Nenhum job perdido além do em curso |
| D3 | 24/10 | Aceitar as sugestões do artigo: GPTQ 3 bits e AWQ no SmolLM3 saem da matriz e são registrados como limitações | Usuário | substituir por outro método; manter falhando | Mecanismo `excluir`; menos jobs inúteis |
| D4 | 24/10 | Rodar primeiro o que cabe (rtn, gptq, awq), DWQ por último | Usuário | esperar memória; fechar Chrome/Claude | Fila deixa de travar na cabeça |
| D5 | 24/10 | Medir o pico de memória da quantização | Usuário | recalibrar por estimativa | Descobriu-se a estimativa **baixa** |
| D6 | 24/10 | Limitar o Qwen3-4B a fp e rtn (GPTQ/AWQ/DWQ fora) | Usuário | parametrizar menos amostras; outro hardware | Reaberto depois (GPTQ: D12 e D13; AWQ: D16) |
| D7 | 24–25/10 | Verificar 3 vezes antes de agir (token irrestrito); trocar por token só de leitura | Usuário | — | Regras de cautela registradas |
| D8 | 25/10 | C agora, A depois, B por último se A for considerável | Usuário | refazer tudo com float32 | A virou diagnóstico (D12) |
| D9 | 25/10 | Preparar protocolo-v2 e análise secundária pós-hoc da H3 | Usuário | só reportar | Rascunho v2; análise log |
| D10 | 25/10 | Manter limiares absolutos; SmolLM3 caso à parte | Usuário | limiares proporcionais | H1/H3 não corroboráveis; sensibilidade registrada |
| D11 | 25/10 | Recalibrar os guardas (fp/rtn e depois GPTQ/AWQ/DWQ) | Usuário | — | Fórmulas em LOGICA §4 |
| D12 | 25/10 | Escrever e validar o GPTQ camada a camada | Usuário | — | Mesmo método, 3–4× menos memória; fp32 só diagnóstico |
| D13 | 25/10 | Rodar o GPTQ do Qwen3-4B por esse caminho | Usuário | deixar excluído | Resultado NaN (informativo) |
| D14 | 25/10 | Inserir o teste do DWQ com alvos entre o GPTQ do 4B e as réplicas de AWQ | Usuário | — | Abortado pela trava (informativo) |
| D15 | 25/10 | Preparar diagnósticos DWQ (sub-lotes) e perfil do AWQ | Usuário | descartar AWQ/DWQ | Causa da memória do AWQ achada |
| D16 | 25/10 | AWQ com liberação por bloco e buscar outras melhorias | Usuário | — | `awq_liberar.py` validado |
| D17 | 25/10 | Documentar cada passo e a lógica alterada | Usuário | — | Este caderno e `LOGICA-QUANTIZACAO.md` |
| D18 | 25/10 | Push direto na `main`; painel mantido com o piloto; `protocolo.md` restaurado à v1 e tag `protocolo-v2`; sessões agrupadas | Usuário | branch + PR; publicar dados parciais; v2 como rascunho; só renomear | Fase 14 |

**Decisões de implementação do Assistente (sem escolha do usuário):** fator 1,4 do pico do MLX ao footprint; teto de partida = metade da RAM; estimativa `2,1 × fp_gb + 0,5` do GPTQ camada a camada; `awq_liberar` sempre ativo na matriz; remoção de entradas de `excluir` quando a alternativa foi validada; marcação de linhas NaN como inválidas (mesma regra que o usuário aprovou para o SmolLM3). Todas estão em `LOGICA-QUANTIZACAO.md` com o motivo.

## 3. E-mails de aviso enviados ao usuário (para o e-mail do autor)

| # | Quando | Assunto |
|---|---|---|
| 1 | 24/10 noite (≈20:18) | Medição de pico de memória da quantização terminou |
| 2 | 24/10 noite | Inspeções leves concluídas (NaN do GPTQ no SmolLM3 e outliers) |
| 3 | 25/10 ≈04h | Causa do NaN do GPTQ no SmolLM3 confirmada + andamento da fila |
| 4 | 25/10 ≈16h20 | Lista automática concluída; NaN do GPTQ explicado; diagnósticos DWQ/AWQ em andamento |
| 5 | 25/10 ≈16h30 | Diagnósticos DWQ/AWQ concluídos: causa do pico do AWQ identificada |

A ordem dos e-mails 1 e 2 na noite de 24/10 não foi registrada com horário.

## 4. Incidentes, erros e correções de curso (transparência)

| # | Quando | O que ocorreu | Correção / efeito |
|---|---|---|---|
| E1 | 23/10 | `kill` do `caffeinate` negado pelo classificador de permissões | O usuário executou o comando; confirmado |
| E2 | 24/10 | Vigia de e-mail baseado em PID terminou na hora (o `pgrep` não achou o processo) | Dados da medição estavam completos e válidos; e-mail enviado em seguida. Vigias seguintes validam o PID antes de armar |
| E3 | 24/10 | O Assistente sugeriu que o guarda de memória do GPTQ era "conservador demais" | **Errado:** a medição mostrou subestimação (14,7 GB medidos contra 10,1 estimados). Corrigido |
| E4 | 24/10 | O Assistente tentou fechar o Ollama; a ação foi cancelada pelo usuário | Não foi forçado; o usuário fechou o app |
| E5 | 25/10 | Teste de memória do DWQ abortado aos 3 s | `--extra` com valor iniciado em `--` confundiu o `argparse`; corrigido com `--extra=` |
| E6 | 25/10 | Teste do DWQ falhou por caminho com espaço ("Application Support") | Alvos movidos para pasta temporária sem espaço |
| E7 | 25/10 | Verificações de equivalência do GPTQ camada a camada foram rodadas **com o GPTQ do Llama ainda em andamento**, contra o combinado ("sozinho") | Sem dano (swap 1,4 GB). **O tempo do 1,7B (466 s) não é comparável** com o do padrão (310 s) |
| E8 | 25/10 | Teste de fumaça do DWQ rodou ao mesmo tempo que o diagnóstico real; a trava do teste de fumaça disparou | Só o teste de fumaça foi afetado; o diagnóstico real rodou sozinho depois |
| E9 | 25/10 | Fator 1,4 do GPTQ superestima modelos grandes (40,2 previstos × 26,06 medidos) | Sem bloqueio (o limiar de partida tem teto); a corrigir com mais pontos |
| E10 | 25/10 | Estimativa inicial de que a H1 seria corroborável com o Qwen3-4B | Obsoleta após o NaN do Llama e do 4B (v2, seção 4.1) |
| E11 | 25/10 | `protocolo.md` (v1) foi editado após a tag (+10 linhas) | **Resolvido (25/10):** conteúdo consolidado na v2 (conferido item a item) e `protocolo.md` restaurado byte a byte ao da tag `protocolo-v1` |
| E12 | 25/10 | Texto do log do `continuar_lista.sh` cita o pid do Llama em vez do script anterior | Cosmético; a ordem de execução está correta |
| E13 | 25/10 | Backups feitos em pasta temporária | Copiados para `experimentos/resultados/backups/` (com SHA-256) |

## 5. Estado dos dados em 25/10/2026, ~17h

- `resultados.csv`: ~69 linhas de jobs (inclui linhas de falha e as invalidadas); GPTQ válido só no Qwen3-0,6B e no 1,7B (8, 4 e 2 bits); GPTQ do SmolLM3 (8/4/2), do Llama (4) e do Qwen3-4B (4): inválidos (NaN). AWQ válido no 0,6B e no 1,7B (4 bits com três sementes); AWQ do Llama e do Qwen3-4B (4 bits): em execução. DWQ válido no 0,6B e 1,7B.
- Hipóteses (vereditos **provisórios** do `analise.py`, 25/10 ~17h, sem gravar `hipoteses.json`): **H1** aguardando o AWQ do Llama e do Qwen3-4B (hoje 2 modelos avaliáveis); **H2** inconclusiva (ordem 0,6B < 1,7B < 4B confirmada em 1 de 2 células do RTN); **H3** inconclusiva sem o SmolLM3 (incrementos médios de L 0,39 → 0,51) e **refutada** com ele (0,46 → 0,42) — as duas versões serão reportadas; **H4** aguardando a fase de velocidade. Análise secundária da H3 (log): superlinear nos 5 modelos no RTN (média de S 9,4; IC 9,1–9,6).

## 6. Pendências

1. AWQ do Llama e do Qwen3-4B (em execução); depois 8, 3 e 2 bits se o resultado de 4 bits for válido.
2. DWQ: repetir o diagnóstico até salvar o modelo, com o cache do MLX limitado; comparar a qualidade com o DWQ padrão.
3. Fase de velocidade (H4), com a máquina ociosa; simular antes (`matriz.py --fase velocidade --simular`), porque uma falha de velocidade grava `falhou:velocidade` na linha de qualidade válida e a análise a descartaria.
4. Repetir o tempo do `Qwen3-0,6B-fp` (8 h 53) para confirmar a hipótese do swap (script não agendado).
5. Desligar o agendamento `launchd` `com.edgeai.etapa4` (continua carregado; um marcador impede a repetição).
6. ~~Consolidar `protocolo-v2`; restaurar `protocolo.md` à tag; commit e tag `protocolo-v2`~~ — feito em 25/10 (Fase 14).
7. ~~Colocar `resultados.csv` sob controle de versão~~ — feito em 25/10 (Fase 14); as logs continuam fora do git (`.gitignore`) e devem ser arquivadas à parte.

## 7. Como reproduzir os pontos principais

```bash
cd edge-ai-quantization
# validação do GPTQ camada a camada (0,6B): gerar e comparar com o padrão
.venv/bin/python experimentos/gptq_camada.py --model Qwen/Qwen3-0.6B --mlx-path /tmp/gptq06 --bits 4 --group-size 64 --seed 0 --hessiana bf16
.venv/bin/python experimentos/comparar_gptq.py /tmp/gptq06 experimentos/modelos/Qwen3-0.6B-gptq-4b-g64-s0 --nll experimentos/resultados/nll/Qwen3-0.6B-gptq-4b-g64-s0.npy
# teste da Hessiana (mecanismo do NaN)
.venv/bin/python experimentos/testar_hessiana.py --modelo meta-llama/Llama-3.2-3B --ate 28 --down todas --cpu
# validação do AWQ com liberação por bloco
.venv/bin/python experimentos/awq_liberar.py --model Qwen/Qwen3-0.6B --mlx-path /tmp/awq06 --bits 4 --group-size 64 --seed 0
.venv/bin/python experimentos/comparar_gptq.py /tmp/awq06 experimentos/modelos/Qwen3-0.6B-awq-4b-g64-s0 --nll experimentos/resultados/nll/Qwen3-0.6B-awq-4b-g64-s0.npy
```
Checksums SHA-256 dos scripts no estado de 25/10 às 16h48: ver `LOGICA-QUANTIZACAO.md` e `git status`; backups em `experimentos/resultados/backups/`.
