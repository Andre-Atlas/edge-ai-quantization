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

### Fase 15 — AWQ dos modelos grandes, correção do `analise.py` e da fórmula de memória (25/10, ~17h–22h)
1. **AWQ 4 bits com `awq_liberar.py`** (agendado 16:45): **Llama-3.2-3B** válido (63 min; PPL 8,180; Q 0,955, IC 0,954–0,957; RTN 0,932; footprint 13,37 GB) e **Qwen3-4B** válido (78 min; PPL 14,662; Q 0,932, IC 0,928–0,935; **abaixo do RTN, 0,942**; footprint 13,18 GB). Primeiros métodos com calibração válidos nos modelos grandes (o GPTQ dava NaN). Fonte: `logs/etapa4.log`, linhas `[conferencia]`.
2. **Erro de código encontrado (Assistente):** o `analise.py` tratava o GPTQ do Qwen3-4B, que já tinha rodado e saído NaN, como *pendente*, e por isso H1 ficava "aguardando dados" para sempre. Um job executado e inválido é um resultado definitivo de falha, não um dado que pode chegar. **Correção:** `carregar()` registra em `INVALIDOS` os jobs com status `falhou:invalido-*`, e `pendente()` não os considera pendentes. Testado com os dados reais e com dois casos sintéticos. **Efeito:** H1 passa de "aguardando dados" para **inconclusiva** (critério atendido em 2 dos 4 modelos contados: Qwen3-1,7B e Llama; Qwen3-0,6B e Qwen3-4B definitivamente não). H3 não muda (inconclusiva sem o SmolLM3, refutada com ele). **Autorizado pelo usuário** por alterar um veredito. Cópia anterior: `resultados/backups/analise.py.antes-da-correcao-invalidos`.
3. **Fórmula de memória do AWQ refeita.** Com o Llama (13,37 GB) a fórmula anterior (`4,0 + 1,4 × fp_gb`, prevista 13,0) subestimou em 0,37 GB, e o Qwen3-4B (13,18 GB) mostrou que o footprint **não cresce com `fp_gb`**. Ajuste nos quatro pontos medidos: um termo em `fp_gb` sai com coeficiente ≈ 0 (−0,11), e o modelo de **um único termo**, o tamanho das entradas guardadas de um bloco (`input_feat` = 128 × 512 tokens × 2 B × (3d + o_in + 2d + i)), explica tudo com resíduo máximo de 0,10 GB: **footprint = 0,5 + 3,66 × input_feat + 0,3 GB** (margem para nunca subestimar). Previsto contra medido: 5,71 × 5,43 · 9,64 × 9,30 · 13,57 × 13,37 · 13,57 × 13,18 (folga mínima 0,2 GB). Quatro pontos para dois parâmetros: sobram dois graus de liberdade, mas é pouco; a 3B e 4B a regra foi validada só nos dois modelos medidos. Cópia anterior: `resultados/backups/matriz.py.antes-da-formula-awq-v2`.
4. **Execução:** `agendar_awq_restantes.sh` (22:10): AWQ 3, 2 e 8 bits do Llama e depois do Qwen3-4B, em série, sozinhos na máquina (6 jobs, ~1 h cada). Conferência de NaN ao fim de cada um.
5. **Estado em 22:06:** nada rodando antes do passo 4; máquina com 12,8 GB livres, swap 1,4 GB, bateria 100%.

### Fase 16 — AWQ de 3, 2 e 8 bits dos modelos grandes; segunda correção da fórmula de memória (26/10, 05h)
1. **Execução concluída (25/10 22:10 → 26/10 05:02, 6 jobs, todos válidos, sem NaN).** Tempos: Llama 3b 61 min, 2b 61 min, 8b 65 min; Qwen3-4B 3b 73 min, 2b 72 min, 8b 80 min. Fonte: `logs/etapa4.log`.
2. **Retenção de qualidade Q (IC 95%), AWQ contra RTN:**

| Largura | Llama-3.2-3B AWQ | Llama RTN | Qwen3-4B AWQ | Qwen3-4B RTN |
|---|---|---|---|---|
| 8 bits | 0,994 | 0,999 | 0,990 | 1,005 |
| 4 bits | **0,955** (0,954–0,957) | 0,932 | 0,932 (0,928–0,935) | **0,942** |
| 3 bits | **0,809** (0,805–0,813) | 0,639 | **0,807** (0,800–0,814) | 0,652 |
| 2 bits | 0,016 (0,009–0,028) | 0,003 | **0,170** (0,164–0,177) | 0,001 |

   A 3 bits o AWQ supera o RTN com folga nos dois modelos (+0,17 e +0,16); a 2 bits o Qwen3-4B é o único caso em que um método com calibração evita o colapso total; a 8 bits o AWQ fica ligeiramente **abaixo** do RTN; a 4 bits há um modelo a favor (Llama) e um contra (Qwen3-4B).
3. **Veredito provisório inalterado** (`analise.py`, 26/10): H1 inconclusiva (2 de 4 modelos contados); H2 inconclusiva; H3 inconclusiva sem o SmolLM3 e refutada com ele (o AWQ não entra na 2ª metade da H3, que usa GPTQ e DWQ); H4 aguardando a velocidade. Análise secundária (log): o AWQ do Qwen3-4B também é superlinear (S = 1,41; IC 1,38–1,45).
4. **Erro de previsão (Assistente):** a fórmula de memória do AWQ, ajustada só em jobs de 4 bits, **subestimou os de 8 bits em 1,24 GB (Llama) e 1,42 GB (Qwen3-4B)** (footprint 14,81 e 14,99 GB contra 13,57 previstos). Os de 3 e 2 bits ficaram dentro (12,99, 12,70, 12,74 e 12,27 GB). Nada travou (a máquina tinha 12,8 GB livres e o limiar de partida tem teto de metade da RAM), mas a regra "nunca subestimar" foi violada.
5. **Fórmula refeita com 10 medições** (4 bits: Qwen3-0,6B e 1,7B; 2, 3, 4 e 8 bits: Llama e Qwen3-4B): `footprint = −1,13 + 3,66 × input_feat + 0,40 × bits + 0,4 GB`. Alternativas testadas: só `input_feat` (resíduo máximo 1,61 GB), `input_feat` + tamanho quantizado (0,40 GB) e `input_feat` + bits (**0,23 GB**, escolhida). O termo de 0,40 GB por bit coincide com o crescimento do modelo quantizado (≈ `fp_gb`/16 por bit). Cobre as 10 medições, folga mínima 0,14 GB. Cópia anterior: `resultados/backups/matriz.py.antes-da-formula-awq-v3`. **Limite:** só dois modelos medidos em várias larguras (e os dois de 3–4B têm a mesma soma de dimensões, 26 624).
6. **Estado da grade v2 (72 jobs de qualidade):** **59 válidos**, 5 inválidos (NaN do GPTQ), **8 a rodar**: GPTQ 8 e 2 bits do Llama e do Qwen3-4B (4 jobs; NaN esperado) e DWQ do SmolLM3 (4 jobs; caso à parte).

### Fase 17 — Fase de velocidade (H4): riscos encontrados, proteções e execução (26/10)
1. **Pergunta do usuário:** quanto tempo leva a fase de velocidade. Estimativa (do código e do piloto, não medida): 1–1,5 h (MLX 40–60 min; K-quants 10–15 min).
2. **Quatro riscos de corrupção de dados encontrados na leitura do código (nenhum chegou a ocorrer):**
   - (a) `rodar.py` gravava `status = completo` na fase de velocidade, o que **desfaria a marca `falhou:invalido-nan`** dos 5 GPTQ com NaN (e a fila da `matriz.py` os incluía: só filtrava "modelo em disco");
   - (b) `matriz.py` gravava `status = falhou:velocidade` na linha de **qualidade válida** se um job de velocidade falhasse (a análise a descartaria);
   - (c) `rodar.py` regravava `data` e `commit` da linha, perdendo **quando e com que versão a qualidade foi medida**;
   - (d) `rodar.py` regravava `impl`, que marcaria como `awq_liberar.py`/`gptq_camada.py` linhas geradas pelo `mlx_lm` padrão.
   Além disso, a velocidade dos K-quants não estava automatizada (o `kquant.py` só documenta o `llama-bench`).
3. **Proteções (plano aprovado):** `rodar.py` — a velocidade só acrescenta as suas colunas (`prefill_tps`, `decode_tps`, `pico_memoria_gb`, `data_velocidade`, `commit_velocidade`), **pula** linhas com `status falhou*` sem medir nem gravar, e só promove `qualidade → completo`; `matriz.py` — fila da velocidade = qualidade **válida** e modelo em disco (59 jobs; exclui os 5 inválidos e os 8 sem qualidade) e falha de velocidade em coluna própria (`erro_velocidade`), sem tocar no `status`; novo `kquant_velocidade.py` (`llama-bench -p 512 -n 128 -r 5 -ngl 99`, retomável, 25 configurações); novo `rodar_velocidade.sh` (backup do CSV, espera, pré-voo que **registra as condições da máquina**, MLX, K-quants, controle de contaminação e checagem de integridade das colunas de qualidade contra o backup).
4. **Testes antes de disparar (em cópia do CSV):** Qwen3-0,6B RTN 4 bits: `qualidade → completo` com decode 404,8 tok/s e prefill 14 143 tok/s (piloto: 402 e 13 745), `data`/`commit`/`impl`/PPL/acurácia/tamanho/versões intactos, nenhuma outra linha alterada; GPTQ do SmolLM3 (inválido): **pulado** sem gravar; `llama-bench` no Qwen3-0,6B Q8_0: decode 276,9 tok/s, retomável (reexecução pula). Fila simulada: 59 jobs, nenhum inválido.
5. **Decisão do usuário (D21):** rodar **agora** e **manter o app Claude aberto** (para análise e envio do e-mail). O app consome CPU, GPU e memória, e a H4 mede GPU e banda de memória: o efeito **não pode ser eliminado, só medido e declarado**. Por isso o orquestrador mede a **mesma configuração** (Qwen3-0,6B RTN 4 bits) 3 vezes antes e 3 depois da fase (sem gravar no CSV) e amostra a cada 30 s o CPU do app Claude e o total do sistema (`logs/velocidade-amostras.log`); `resumo_contaminacao.py` calcula a deriva e o CV dos controles. Se a deriva for pequena e o app ficar quase parado, o resultado vale com a ressalva declarada; senão, os jobs afetados são refeitos com o app fechado. Durante a execução o Assistente não faz comandos pesados (só o vigia de 1 h, que consome milissegundos por minuto).
6. **Segundo erro (Assistente), pego antes de rodar:** o pré-voo exigia 14 GB livres e o guarda da `matriz.py` usava a regra da **qualidade** (12,8 GB para o Qwen3-4B fp), mas com o app aberto havia 11,3 GB livres: a fase ficaria horas esperando sem necessidade, porque a velocidade não carrega logits nem calibração. O orquestrador foi cancelado ainda na espera de 5 min (nada medido; `resultados.csv` idêntico ao backup), e corrigido: pré-voo ≥ 10 GB e, na fase de velocidade, limiar da matriz = peso do modelo (fp ou quantizado, `fp_gb × (bits + 0,5)/16`) + 2,5 GB (Qwen3-4B fp: 10,6 GB). Commit refeito antes de relançar.
7. **Erro menor (Assistente):** o primeiro patch do orquestrador falhou por conflito de delimitadores de heredoc no meu próprio comando; não alterou nada (o arquivo estava intacto) e foi refeito pela ferramenta de edição.

### Fase 18 — Fase de velocidade executada (26/10, 10:40–11:00)
1. **Execução** (`rodar_velocidade.sh`, 10:40 → 11:00, **20 min**, bem abaixo da estimativa de 1–1,5 h): pré-voo "ociosa" (carga 1,4; 11,5 GB livres; swap 1,0 GB; bateria 100%; **apps abertos: Calendar e o app Claude**, entre outros); **59 jobs MLX** em 9,5 min e **25 configurações K-quants** em 3 min; 0 falhas; `[INTEGRIDADE] OK` (0 diferenças nas colunas de qualidade e nos status; as 5 linhas inválidas continuam `falhou:invalido-nan`). Fonte: `logs/etapa4.log`, `resultados/resultados.csv`, `resultados/kquant-velocidade.csv`.
2. **Contaminação medida (app Claude aberto por decisão do usuário, D21):** CPU do app Claude em 26 amostras de 30 s: **média 62,5%, máximo 219%, acima de 10% em 23 das 26** — o app esteve ativo; CPU total do sistema média 70,5%. O **controle** (Qwen3-0,6B RTN 4 bits, 3 medidas antes e 3 depois): decode **409,6 tok/s (CV 0,2%) antes e 407,1 (CV 0,1%) depois, deriva −0,6%**. Leitura: o ruído entre medidas repetidas é pequeno, mas o app **esteve** consumindo CPU durante a fase; os controles medem só o início e o fim (não a fase inteira) e só a configuração menor. **Ressalva declarada:** as velocidades absolutas podem estar levemente subestimadas. *(Corrigido na Fase 19, item 3: a CPU atribuída ao app incluía o próprio benchmark; a rodada 2, com o app fechado, mostrou diferença média de −0,4%.)*
3. **Resultado da H4 (provisório, `analise.py`):** eficiência `E = (ganho de decode) / (redução de tamanho)` a 4 bits: **SmolLM3-3B 0,85 (RTN), Llama-3.2-3B 0,88 (RTN e AWQ), Qwen3-4B 0,87 (RTN e AWQ)** — todos acima do limiar de 0,6 do protocolo: **H4 corroborada (provisória)**. Mesmo se a contaminação reduzisse o decode em 20%, E ficaria em ~0,7, ainda acima do limiar; a margem é grande. O SmolLM3 só tem o RTN (GPTQ inválido; AWQ inexistente; DWQ não executado).
4. **Padrão por tamanho (não previsto):** E cresce com o modelo: Qwen3-0,6B **0,60–0,63**, Qwen3-1,7B **0,81–0,82**, modelos de 3–4B **0,85–0,88**. Nos modelos pequenos, uma parte maior do tempo de decodificação não depende dos pesos. Decodificação a 4 bits: 2,1× (0,6B), 2,8× (1,7B), 3,0–3,1× (3–4B) mais rápida que bf16, com tamanho 3,5× menor.
5. **K-quants (llama.cpp, decode em tok/s; F16 → Q4_K_M → Q2_K):** Qwen3-0,6B 182,6 → 348,3 → 357,8; Qwen3-1,7B 77,4 → 187,4 → 209,1; Llama-3.2-3B 43,7 → 116,8 → 135,8; SmolLM3-3B 45,0 → 119,3 → 141,1; Qwen3-4B 34,8 → 94,0 → 106,7. Q3_K_M não é mais rápido que Q4_K_M (em alguns modelos é ligeiramente mais lento). Prefill praticamente igual entre os tipos.
6. **Vereditos provisórios finais desta rodada:** **H4 corroborada**; H1, H2 e H3 inalteradas (inconclusivas; H3 refutada com o SmolLM3 contado).

### Fase 19 — Rodada 2 da velocidade, com o app Claude fechado (26/10, 12:59–13:13)
1. **Por que:** a rodada 1 teve o app aberto (D22: o usuário repetiu a medida para eliminar a ressalva). A primeira tentativa (`nohup` a partir da sessão) morreu durante a espera de 5 min, quando o app foi fechado, **sem alterar nenhum dado** (E18). Relançada como job de uso único no `launchd` (`com.edgeai.velocidade-r2`), independente do app. A rodada 1 foi arquivada em `resultados/velocidade-rodada1-app-aberto.csv`, `resultados/kquant-velocidade-rodada1-app-aberto.csv` e `resultados/backups/`.
2. **Condições:** o pré-voo da rodada 2 listou os apps abertos **sem o Claude** (rodada 1: com o Claude); 13,7 GB livres; swap 0,3 GB; bateria 100%. 59 jobs MLX + 25 K-quants, 0 falhas, `[INTEGRIDADE] OK`. A rodada 2 é a de **referência**; a 1 é a comparação.
3. **Correção da medida de contaminação (E19):** o amostrador de CPU filtrava os processos pelo nome `Claude`, mas o **caminho do ambiente** (`.../Application Support/Claude/.../.venv/bin/python`) também contém "Claude". A coluna `claude_cpu` somava, portanto, **o próprio benchmark** (iguala `total_cpu` nas amostras) e não o app. Os números "app Claude 62,5% (rodada 1)" e "73,8% (rodada 2)" **não medem o app e foram descartados**; a evidência válida de que o app estava aberto/fechado é a **lista de apps do pré-voo**. Filtro corrigido para `/Applications/Claude.app/` em `rodar_velocidade.sh`. A Fase 18 (item 2) fica corrigida por esta nota.
4. **Comparação rodada 2 / rodada 1 (59 jobs MLX pareados):** decode **média −0,4%, mediana −0,1%**, quase todos dentro de ±2,5%; prefill média +0,7%. Maiores diferenças: Llama-3.2-3B RTN 4 bits decode −5,3%; Qwen3-1.7B GPTQ 2 bits prefill −14%. Pico de memória idêntico (|dif| ≤ 0,05 GB). Controles (Qwen3-0,6B RTN 4 bits): rodada 1 409,6 → 407,1; rodada 2 403,9 (CV 1,6%) → 407,2. **Leitura: o app aberto não produziu efeito detectável nas velocidades do MLX** (se produzisse, a rodada 2 seria mais rápida de modo sistemático; não é).
5. **H4 com a rodada de referência:** E (4 bits) = **SmolLM3 0,83; Llama 0,83–0,88; Qwen3-4B 0,87**; Qwen3-1,7B 0,80–0,82; Qwen3-0,6B 0,59–0,64. `analise.py`: **H4 corroborada** (E médio 0,83 / 0,85 / 0,87 nos modelos de 3–4B, todos ≥ 0,6). Veredito igual ao da rodada 1.
6. **K-quants:** mediana −0,4%, mas o **Qwen3-0,6B ficou 17–24% mais lento** na rodada 2 (Q8_0 276→231; Q4_K_M 348→274; Q2_K 358→270) e o Qwen3-1,7B Q3_K_M/Q2_K −7%/−9%. Como a rodada **sem** o app foi a mais lenta, a causa **não** é o app; não identificada (hipóteses não testadas: estado térmico/de energia da GPU, o `llama-bench` ser sensível ao ambiente em modelos pequenos). Os valores K-quants do Qwen3-0,6B têm, portanto, incerteza de ~20% entre rodadas; as conclusões sobre modelos de 3–4B (diferença ≤ 1%) não são afetadas. Fica registrado como limitação; as duas rodadas estão arquivadas.
7. **Anomalia (existia nas duas rodadas):** Llama-3.2-3B AWQ 2 bits mediu decode **46 760 / 49 294 tok/s** (a ordem de 1000× o esperado): provavelmente gera 0–poucos tokens (modelo degenerado), de modo que o valor **não é uma velocidade válida**. Está fora da H4 (que usa 4 bits), mas **deve ser excluído/anotado** em qualquer tabela de velocidade; checar `avaliar.velocidade` (parada antecipada por EOS) antes da redação.

### Fase 20 — Anomalia do decode do Llama AWQ 2 bits e GPTQ 8/2 bits restantes (26/10, 13:30–)
1. **Causa da anomalia (diagnosticada):** o `Llama-3.2-3B-awq-2b` emite o token de fim (EOS) **logo após 1 token** (`generation_tokens = 1`, `finish_reason = stop`), então `generation_tps` é calculado sobre 1 token (~40–49 mil tok/s). Os controles `rtn-2b` e `awq-4b` geram os 128 tokens (172 e 122 tok/s). Ou seja, o modelo 2-bit do AWQ está **degenerado**, coerente com a sua qualidade; o valor **não é velocidade**. Os outros 58 jobs têm decode ≤ ~480 tok/s, compatível com 128 tokens.
2. **Ação:** (a) `avaliar.velocidade` passa a devolver o mínimo de tokens gerados (`decode_tokens_min`) para sinalizar linhas assim nas próximas medições; (b) a linha existente recebeu `decode_tokens_min = 1` e `obs_velocidade = INVALIDA…`, **mantendo o dado bruto**; (c) o valor não entra em nenhuma tabela de velocidade (a H4 usa 4 bits e não é afetada). *Alternativa descartada:* proibir o EOS na medição, que mudaria o método das 58 linhas válidas.
3. **GPTQ 8 e 2 bits (Llama-3.2-3B e Qwen3-4B), 4 jobs:** agendados em série, camada a camada, como job de uso único no `launchd` (`agendar_gptq_restantes.sh`, iniciado 13:35). Expectativa: NaN, pela mesma causa do 4 bits (Hessiana em bf16); o objetivo é **registrar o resultado**, válido ou não. `conferir_nan.py` marca automaticamente como `falhou:invalido-nan` os que derem NaN (backup antes). Em seguida, a velocidade dos que forem válidos (se houver) roda numa rodada à parte, com a máquina ociosa.

### Fase 21 — DWQ do SmolLM3 cancelado e excluído (26/10, 13:50)
1. **Decisão (D24, usuário):** a tentativa do DWQ 4 bits do SmolLM3 (agendada 13:39 para depois dos GPTQ, com trava de 1,5 GB) foi **cancelada antes de iniciar**: o usuário a considerou desnecessária. Job do `launchd` removido; nada rodou nem foi gravado.
2. **Por que o DWQ não admite o corte por camada (raciocínio registrado):** GPTQ e AWQ calibram cada camada isoladamente; o DWQ treina escalas e vieses de ponta a ponta contra os logits do modelo original, então todas as camadas, o modelo original (ou alvos) e os logits de 128 mil tokens precisam estar disponíveis. Um DWQ por blocos seria outro método (BRECQ/OmniQuant), fora do objeto do estudo. Caminhos não testados e não necessários agora: alvos em disco (`--target-dir`), lote/sequência menores e limite de cache do MLX (o footprint do 0,6B foi 20 GB contra 5,9 GB de pico do MLX).
3. **Registro:** `matriz.json` → `excluir` ganhou SmolLM3 × DWQ (motivo de memória). Efeito na análise: o SmolLM3 passa a "não avaliável" para o DWQ (como DWQ do Llama e do Qwen3-4B), sem contar contra o método. A grade de qualidade fica em 72 jobs, dos quais 4 pendentes (GPTQ 8/2 bits do Llama e do Qwen3-4B, em execução).

### Fase 22 — GPTQ 8 e 2 bits concluídos; matriz de qualidade fechada (26/10, 13:35–18:33)
1. **Resultado:** os 4 jobs da fila (`agendar_gptq_restantes.sh`) e a repetição do Llama de 8 bits (`repetir_llama_gptq8.sh`) terminaram. **Todos deram perplexidade NaN**: Llama-3.2-3B 8, 4 e 2 bits; Qwen3-4B 8, 4 e 2 bits. `conferir_nan.py` os marcou como `falhou:invalido-nan` (backups antes). Padrão completo: o **GPTQ do mlx-lm 0.32.2 (Hessiana em bf16) é inválido em todos os modelos de 3–4B e em todas as larguras testadas** (SmolLM3 8/4/2, Llama 8/4/2, Qwen3-4B 8/4/2); só o Qwen3-0,6B e o 1,7B dão modelos válidos.
2. **Falha isolada:** o Llama de 8 bits falhou na 1ª tentativa (processo encerrado de forma anormal após as 28 camadas, sem erro no log; pico 16,8 GB; suspeita de pressão de memória pela coincidência com outras atividades, **não confirmada**). Repetido sozinho: concluído em 4004 s, com NaN como os demais. A 1ª tentativa fica registrada aqui como falha de execução, não de método.
3. **Grade final da qualidade:** **68 jobs executados** (a grade de 72 menos 4 do DWQ × SmolLM3 excluídos, D24): **59 válidos**, **9 inválidos** (GPTQ NaN), 0 pendentes. Vereditos provisórios inalterados: H1 inconclusiva, H2 inconclusiva, H3 inconclusiva sem o SmolLM3 (refutada com ele), H4 corroborada. `analise.py` regravou `hipoteses.json` — é **provisório**, e o final sai só após as lacunas da etapa 5.
4. **Etapa 4 encerrada na parte de execução.** Restam: repetir o tempo do `Qwen3-0.6B-fp`, desligar o `com.edgeai.etapa4`, arquivar logs e fechar a documentação.

### Fase 23 — Protocolo v3: extensão pré-registrada para tornar H1–H3 decidíveis (26/10, 21:45–23:00)
1. **Pergunta do usuário:** o que fazer para que as outras hipóteses tenham resultado conclusivo, seja ele corroboração ou refutação?
2. **Diagnóstico (raciocínio):** com os dados da v2, nenhum dado ainda possível muda H1, H2 ou H3.
   - **H1:** 2 sucessos e 2 "nenhum ciente" fixos; a corroboração exige 4 de 4 e a refutação exige 3.
   - **H2:** só 2 células, então 50% é inconclusivo por construção.
   - **H3:** a 2ª metade tem 2 modelos avaliáveis.
   - Mudar as regras olhando os dados seria ajuste pós-hoc. Daí a decisão por uma **v3 pré-registrada**, com dados novos e regras decidíveis.
3. **Decisões do usuário (D25):**
   - **Frentes escolhidas:** novos modelos, GPTQ corrigido (fp32) e regras mais precisas. A reanálise secundária dos dados da v2 não foi escolhida.
   - **Restrições:** execuções só de sexta 18h a segunda (dia inteiro); artigo até **18/12/2026**.
4. **O que foi feito (dias úteis, sem execução de modelo):**
   - `protocolo-v3.md` escrito.
   - `empacotar()` de 3 bits no `gptq_camada.py`, conferido bit a bit.
   - Método `gptq_fp32` no `rodar.py`.
   - `--config`, grades por bloco e `--ate` no `matriz.py`.
   - `matriz-v3.json` com 52 jobs.
   - `analise_v3.py` e `testes_v3.py`: 23 de 23 aprovados. Os testes acharam 2 problemas, os dois corrigidos antes do commit:
     - um erro nos dados sintéticos de teste;
     - empates de arredondamento no 6 bits do `mx.quantize`. Daí o teste de formato com inteiros, separado do teste com valores reais.
   - `rodar_fim_de_semana.sh` e o job semanal.
   - Download dos Qwen2.5 0.5B, 1.5B e 3B (12 GB).
   - O **Llama-3.2-1B depende de acesso**, pedido pelo usuário na página do Hugging Face. Enquanto não houver acesso, o orquestrador o pula.
5. **Alternativas descartadas:**
   - **Limiares proporcionais aplicados aos dados da v2:** seria pós-hoc.
   - **Gemma-3-1B:** tem janela deslizante, que o `gptq_camada.py` não suporta.
   - **Proibir o EOS** na medida de velocidade: não se aplica aqui.
   - **8 bits na v3:** não entra em nenhuma hipótese e custaria ~25% a mais de tempo de máquina.
5b. **Teste do `analise_v3.py` nos dados da v2, antes da tag:** só para conferir o funcionamento. Foi declarado no protocolo-v3, seção 3, "Amostra e transparência". Resultado: σ_cal = 0,0065; H1 com 1/2/1 entre 4 avaliáveis; H2 com 1 de 4 células válidas; 1ª metade da H3 superlinear em 5 de 5. Tudo "não avaliável" por N insuficiente. Nenhuma regra foi mudada depois disso.
6. **Tag `protocolo-v3`** criada neste commit, **antes** de qualquer execução da extensão. A 1ª janela é 30/10, 18h.
6b. **Llama-3.2-1B baixado (26/10, depois da tag):** o usuário já tinha acesso. O `config.json` confirma as dimensões da `matriz-v3.json`: d 2048, i 8192, n 16, o_in 2048, embedding amarrado, 2,47 GB. O `matriz-v3.json` não precisou mudar (a nota "a conferir" fica como registro). Os 4 modelos novos estão baixados e entram na fila do fim de semana 1.
7. **Job diário `com.edgeai.etapa4` desligado:** estava pendente desde a etapa 4; o plist foi renomeado para `.desativado`.

### Fase 24 — Extensão v3: fins de semana de 30/10 e 06/11, incidente de energia e pausa (30/10–09/11)
1. **Execução (job `com.edgeai.v3-fds`, sexta 18h):** nos dois fins de semana o orquestrador rodou a validação de regressão (`gptq_camada.py` com Hessiana bf16 contra o GPTQ padrão do Qwen3-0,6B): **704 de 704 tensores idênticos** nas duas vezes. A validação completa (sanidade RTN3 > GPTQ3 > GPTQ4) ainda não fechou, porque o GPTQ-fp32 do 0,6B a 3 bits não terminou a avaliação.
2. **Incidente de 30/10 (energia):** o Mac estava na bateria (48%) e entrou em repouso várias vezes. O `caffeinate -i -s` só impede o repouso na tomada (`-s`), então a fila travou na avaliação do GPTQ-fp32 do 0,6B a 4 bits. Diagnóstico por `pmset -g log`. **Causa raiz:** o orquestrador não exigia a tomada. Não foi corrigido em código; a proposta (trava de tomada no `rodar_fim_de_semana.sh`) ficou pendente de decisão do usuário.
3. **Fim de semana de 06/11:** a avaliação do GPTQ-fp32 de 4 bits no 0,6B terminou (`tempo_quant_s` 3345 s; **PPL 22,58**, contra 24,33 do RTN, 23,72 do AWQ e 22,90 do GPTQ padrão, todos a 4 bits no mesmo modelo; leitura descritiva, de uma execução, sem inferência). A quantização de 3 bits foi gerada (4459 s), mas a avaliação não terminou. O registro `tempo` da avaliação de 4 bits (85 749 s) inclui as horas em repouso e **não deve ser usado como medida de tempo**.
4. **Pausa (08/11, 20h44):** a pedido do usuário ("pare até segunda ordem"), o job foi descarregado (`launchctl bootout`), o plist renomeado para `com.edgeai.v3-fds.plist.pausado` e os processos encerrados (a árvore estava parada, 0% de CPU, havia dias). Nenhum dado foi alterado.
5. **Estado dos dados:** 2 linhas novas no `resultados.csv` (GPTQ-fp32 0,6B 4 bits completa; 3 bits em status `quantizado`, sem avaliação). As NLL por janela de ambas estão em `resultados/nll/`. Nenhuma linha da v2 mudou (comparação com os backups `antes-v3-fds-*`). Os vereditos da v2 permanecem.
6. **Impacto no cronograma:** o plano previa dados congelados em 16/11. Com 1 de 52 jobs concluídos e a retomada dependendo de nova ordem, a v3 completa não cabe nos fins de semana restantes antes de 18/12 sem decisão do usuário (reduzir a grade, ou aceitar H1–H3 como inconclusivas e reportar a v3 como parcial). **Decisão em aberto.**
7. **Painel (9/11):** `exportar_site.py` regenerado (80 linhas; exclui linhas só quantizadas, sem PPL); o painel passou a mostrar o estado do estudo, o método GPTQ-fp32 e os vereditos provisórios da v2.

### Fase 25 — Migração do workspace, fechamento analítico da Etapa 5, 27 fichas e manuscrito (09/11, 19:20–20:05)
1. **Migração do workspace (D26):** o repositório completo (163,5 GB, incluindo `experimentos/modelos/`, `experimentos/ferramentas/` com `llama.cpp` compilado e GGUFs, `experimentos/logs/`, `.venv/` e histórico Git em `3691822`) foi movido do diretório temporário do Claude Desktop (`scratch-workspaces`) para `/Users/andreatlas/Downloads/Edge AI & Quantization`, mantendo um symlink de compatibilidade no caminho antigo e atualizando os caminhos nos plists em `~/Library/LaunchAgents/`.
2. **Validação de sanidade 3 bits do `gptq_fp32` (D27, `protocolo-v3` §4.2):** concluída a avaliação do modelo `Qwen3-0.6B-gptq_fp32-3b-g64-s0` (quantizado em 06/11): **PPL = 40,20** ($Q = 0,5216$), satisfazendo a desigualdade estrita pré-registrada $\text{PPL}_{\text{gptq\_fp32 4b}} (22,58) < \text{PPL}_{\text{gptq\_fp32 3b}} (40,20) < \text{PPL}_{\text{rtn 3b}} (62,26)$. Criado também o orquestrador `experimentos/rodar_bloco1_v3.sh` com trava obrigatória de tomada (`AC Power` via `pmset -g batt`) para o Bloco 1 da v3 (5 modelos da v2 em `gptq_fp32` 4 e 3 bits).
3. **Fechamento analítico da Etapa 5 (D28, `experimentos/analise.py` e `testes_v3.py`):**
   - Filtro explícito em `analise.carregar()` para os 5 modelos e métodos da `v2` (`fp`, `rtn`, `gptq`, `awq`, `dwq`), preservando `n_configuracoes = 59` e a fronteira de Pareto da `v2` sem contaminação pelas linhas `gptq_fp32` da `v3`.
   - **Concordância $Q \times Q_{\text{acc}}$ (`concordancia_acuracia`):** nas $N = 50$ configurações válidas quantizadas da `v2` (`seed == 0`), correlação linear de Pearson **$r = 0,9520$**, correlação de postos de Spearman **$\rho = 0,9413$**, e coincidência do melhor método calibrado a 4 bits em **3 de 4 modelos avaliáveis** (`Qwen3-1,7B`, `Llama-3.2-3B` e `Qwen3-4B`; divergência apenas em `Qwen3-0,6B` onde `gptq` lidera em PPL e `dwq` em acurácia por +0,0063).
   - **Variância entre sementes de calibração do AWQ 4 bits (`variancia_replicas_awq`, sementes 0, 1, 2):** $\sigma_{\text{cal}}(Q) = 0,00649$ combinado ($\text{Qwen3-0,6B}$: $\sigma(Q) = 0,00662$, $\text{CV}_{\text{PPL}} = 0,75\%$; $\text{Qwen3-1,7B}$: $\sigma(Q) = 0,00635$, $\text{CV}_{\text{PPL}} = 0,65\%$).
   - Suite de testes `experimentos/testes_v3.py` ampliada para 25 asserções (`0 falha(s)`).
4. **Etapa 7A — Fichamento bibliográfico completo (`literatura/fichas/`):** redigidas as 27 fichas de leitura estruturadas (`01-shuvo2023efficient.md` a `27-kawrakow2023kquants.md`) e o índice `literatura/fichas/README.md`, seguindo `metodologia/ficha-modelo.md` (Severino + Popper), com marcação explícita de quais textos tiveram PDF integral inspecionado e quais foram fichados a partir de metadados/resumo verificados.
5. **Etapa 7B — Tabelas, figuras e manuscrito científico (`artigo/`):** criado `experimentos/gerar_tabelas_artigo.py` gerando reprodutivelmente 5 tabelas em Markdown e LaTeX (`artigo/tabelas/tabela1` a `tabela5`) e 4 figuras vetoriais SVG (`artigo/figuras/figura1` a `figura4`), além da redação integral do manuscrito científico em português (`artigo/manuscrito.md`, padrão ABNT / Severino + Popper).


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

### Fase 15 — AWQ dos modelos grandes e correções
| Ação | Por quê | Alternativa descartada |
|---|---|---|
| Pedir autorização antes de corrigir o `analise.py` | A correção muda um veredito (H1: aguardando → inconclusiva); decisões que mudam o estudo são do usuário (P5) | Corrigir e informar depois |
| Tratar "executado e inválido" como resultado definitivo | O job já rodou: esperar por ele seria esperar para sempre. O dado bruto continua no CSV (P3) | Reexecutar o GPTQ do 4B |
| Refazer a fórmula com um termo de arquitetura em vez de ajustar uma reta em `fp_gb` | A reta errou o Llama (+1,28 GB) e o Qwen3-4B (−0,97 GB); o termo físico (entradas de um bloco) dá resíduo de 0,10 GB com menos parâmetros | Aumentar a margem da reta |
| Margem de 0,3 GB | A regra existe para nunca subestimar (P2); 0,3 cobre o maior resíduo positivo (0,10) com folga | Margem zero |
| Rodar 3 e 2 bits antes de 8 bits | A curva qualidade × bits e a análise de superlinearidade precisam de 4, 3 e 2; 8 bits é o menos informativo | Ordem da matriz (8, 3, 2) |
| Um job falho não interrompe os seguintes | Uma falha pontual (por exemplo, NaN) não deve parar horas de fila (P3) | Parar no primeiro erro |

### Fase 16 — AWQ de 3, 2 e 8 bits e fórmula de memória
| Ação | Por quê | Alternativa descartada |
|---|---|---|
| Comparar três formas da fórmula antes de escolher | Escolher pela menor resíduo máximo evita ajustar à intuição; a forma com bits (0,23 GB) venceu a com tamanho quantizado (0,40 GB) | Aceitar a primeira que cobrisse |
| Manter a margem de 0,4 GB mesmo com resíduo de 0,23 GB | A regra existe para nunca subestimar (P2) e só há dois modelos medidos em várias larguras | Margem mínima de 0,25 |
| Registrar a violação da regra (E16) | Os 8 bits foram 1,2–1,4 GB acima do previsto: um erro do Assistente que não causou dano, mas que um registro só de acertos esconderia | Omitir |
| Não rodar mais jobs sem antes recalibrar | O próximo job a rodar usaria a fórmula errada para o limiar de partida | Rodar em seguida |

### Fase 17 — fase de velocidade
| Ação | Por quê | Alternativa descartada |
|---|---|---|
| Ler o código antes de rodar | A fase grava no mesmo CSV que guarda a qualidade e as marcas de invalidade; um erro ali corromperia dados já obtidos e irreproduzíveis em horas (P3) | Rodar e corrigir depois |
| Velocidade só **acrescenta** colunas | Preserva a rastreabilidade (`data`, `commit`, `impl`) da medida de qualidade | Regravar a linha inteira (comportamento anterior) |
| Pular linhas inválidas **sem medir** | Medir velocidade de um modelo com NaN não tem significado científico e desfaria a marca de invalidade | Medir e marcar depois |
| Testar na **cópia** do CSV antes de disparar | Se a proteção estivesse errada, o dano ficaria na cópia (P3) | Testar direto no CSV |
| Commit antes da execução | `commit_velocidade` grava o hash do código; com alterações sem commit sairia `+sujo` e a medida não seria reprodutível | Commitar depois |
| Controle antes/depois + amostragem de CPU em vez de pedir silêncio impossível | O usuário decidiu manter o app aberto; um efeito que não se pode eliminar deve ser **medido e declarado** (P1: nada escondido) | Ignorar o app; ou recusar rodar |
| Manter a decisão do usuário de rodar agora | Decisões que mudam o estudo são dele (P5); o Assistente registrou o risco e a mitigação | Insistir em fechar o app |

### Fase 18 — leitura dos resultados da velocidade
| Ação | Por quê | Alternativa descartada |
|---|---|---|
| Reportar a contaminação (app com CPU média de 62%) mesmo com deriva de −0,6% | Os controles medem só início e fim e só o modelo menor; esconder o consumo do app seria omitir um grau de incerteza (P1) | Declarar "máquina ociosa" porque o pré-voo passou |
| Testar se o veredito sobrevive a uma contaminação de 20% | A conclusão deve ser robusta ao pior caso plausível; E ≥ 0,7 mesmo assim | Refazer tudo com o app fechado sem necessidade |
| Registrar o padrão E × tamanho como achado | Não estava no protocolo e é relevante para a discussão: o ganho de velocidade só se aproxima do teórico em modelos maiores | Reportar só o veredito da H4 |

### Fase 25 — migração, fechamento analítico da Etapa 5, fichas e manuscrito
| Ação | Por quê | Alternativa descartada |
|---|---|---|
| Mover os 163,5 GB para `Downloads/Edge AI & Quantization` e criar symlink no caminho antigo | Preserva o trabalho fora do diretório temporário do Claude Desktop sem quebrar scripts antigos | Clonar só o Git e perder 162 GB de modelos/ferramentas |
| Filtrar `analise.carregar()` estritamente aos modelos e métodos da `v2` | Evita que linhas `gptq_fp32` da `v3` alterem `n_configuracoes` (`59`) ou a fronteira de Pareto da `v2` (P1) | Deixar `analise.py` ler todas as linhas do CSV |
| Incluir $Q \times Q_{\text{acc}}$ e $\sigma_{\text{cal}}$ como diagnósticos secundários em `hipoteses.json` | Fecha as lacunas apontadas desde 22/10 sem alterar os vereditos primários pré-registrados (P1) | Alterar os critérios de H1–H4 pós-hoc |
| Marcar em cada ficha se o texto integral (PDF) foi lido ou se o fichamento parte de metadados/resumo verificados | Transparência metodológica exigida por Severino: nunca fabricar páginas ou citações diretas | Omitir a distinção ou inventar páginas |
| Gerar todas as tabelas e figuras por script (`gerar_tabelas_artigo.py`) a partir dos CSVs/JSONs | Garante reprodutibilidade exata e zero transcrição manual entre dados e manuscrito | Montar tabelas à mão |

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
| D19 | 25/10 | Corrigir o `analise.py` (invalidados não são pendentes) e a fórmula de memória do AWQ; rodar os AWQ de 3, 2 e 8 bits do Llama e do Qwen3-4B | Usuário | só relatar; esperar o resultado do 4B para recalibrar | H1 provisória passa a inconclusiva; seis jobs em execução |
| D20 | 25/10 | Avisar por e-mail quando os seis AWQ terminarem | Usuário | — | E-mail 6 (26/10, ~05h) |
| D21 | 26/10 | Rodar a fase de velocidade **agora**, com o app Claude **aberto** (para análise e e-mail) | Usuário | rodar à noite com o app fechado; às 2h | Contaminação medida (controles e amostras), não eliminada |
| D22 | 26/10 | Repetir a fase de velocidade com o app Claude fechado (rodada 2 de referência) | Usuário | manter só a rodada 1 | Diferença média de −0,4%; H4 confirmada sem ressalva |
| D24 | 26/10 | Cancelar e excluir o DWQ do SmolLM3 da matriz | Usuário | tentar rodar com trava de 1,5 GB | Grade da v2 fechada em 68 jobs executados (59 válidos, 9 inválidos) |
| D25 | 26/10 | Criar a extensão pré-registrada `protocolo-v3` (novos modelos + `gptq_fp32` + regras decidíveis) | Usuário | reanálise pós-hoc da v2 | Tag `protocolo-v3`, `matriz-v3.json` (52 jobs) e `analise_v3.py` |
| D26 | 09/11 | Migrar o repositório completo (163,5 GB) para `/Users/andreatlas/Downloads/Edge AI & Quantization` com symlink no caminho antigo | Usuário | clonar só o Git; manter no `scratch-workspaces` | Workspace unificado e preservado |
| D27 | 09/11 | Focar a Etapa 4b (`v3`) nos 5 modelos da `v2` com `gptq_fp32` (concluir avaliação 3b do 0,6B + script com trava `AC Power`) | Usuário | retomar os 52 jobs completos; congelar sem avaliar o 3b | Validação de sanidade 3b concluída; `rodar_bloco1_v3.sh` pronto |
| D28 | 09/11 | Executar fechamento analítico da Etapa 5 ($Q \times Q_{\text{acc}}$ e $\sigma_{\text{cal}}$), 27 fichas de leitura e manuscrito científico completo com tabelas/figuras | Usuário | fazer apenas uma das frentes por vez | Etapas 5, 6 e 7 concluídas de ponta a ponta |

**Decisões de implementação do Assistente (sem escolha do usuário):** fator 1,4 do pico do MLX ao footprint; teto de partida = metade da RAM; estimativa `2,1 × fp_gb + 0,5` do GPTQ camada a camada; `awq_liberar` sempre ativo na matriz; remoção de entradas de `excluir` quando a alternativa foi validada; marcação de linhas NaN como inválidas (mesma regra que o usuário aprovou para o SmolLM3). Todas estão em `LOGICA-QUANTIZACAO.md` com o motivo.

## 3. E-mails de aviso enviados ao usuário (para o e-mail do autor)

| # | Quando | Assunto |
|---|---|---|
| 1 | 24/10 noite (≈20:18) | Medição de pico de memória da quantização terminou |
| 2 | 24/10 noite | Inspeções leves concluídas (NaN do GPTQ no SmolLM3 e outliers) |
| 3 | 25/10 ≈04h | Causa do NaN do GPTQ no SmolLM3 confirmada + andamento da fila |
| 4 | 25/10 ≈16h20 | Lista automática concluída; NaN do GPTQ explicado; diagnósticos DWQ/AWQ em andamento |
| 5 | 25/10 ≈16h30 | Diagnósticos DWQ/AWQ concluídos: causa do pico do AWQ identificada |
| 6 | 26/10 ≈05h | AWQ de 3, 2 e 8 bits do Llama e do Qwen3-4B concluídos |
| 7 | 26/10 ≈11h | Fase de velocidade concluída; H4 corroborada (provisória) |

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
| E14 | 25/10 | `analise.py` tratava job executado e inválido (NaN) como pendente: H1 "aguardando dados" indefinidamente | Corrigido com autorização do usuário (Fase 15); mudou o veredito provisório da H1 |
| E15 | 25/10 | Fórmula de memória do AWQ (ajuste linear em `fp_gb`) subestimou o Llama em 0,37 GB | Refeita com termo de arquitetura (Fase 15); a regra "nunca subestimar" volta a valer nos 4 pontos |
| E16 | 26/10 | A fórmula de memória do AWQ (ajustada só em 4 bits) subestimou os jobs de 8 bits em 1,24 e 1,42 GB | Refeita com a largura de bits (Fase 16); sem dano (limiar de partida com teto; 12,8 GB livres) |
| E17 | 26/10 | Limiares de memória da fase de velocidade copiados da fase de qualidade (12,8 GB para o Qwen3-4B fp; pré-voo de 14 GB) com 11,3 GB livres: a fase esperaria horas | Pego antes de rodar; orquestrador cancelado na espera; limiar da velocidade = pesos + 2,5 GB; pré-voo de 10 GB (Fase 17) |

## 5. Estado dos dados em 25/10/2026, ~17h

- `resultados.csv`: ~69 linhas de jobs (inclui linhas de falha e as invalidadas); GPTQ válido só no Qwen3-0,6B e no 1,7B (8, 4 e 2 bits); GPTQ do SmolLM3 (8/4/2), do Llama (4) e do Qwen3-4B (4): inválidos (NaN). AWQ válido no 0,6B e no 1,7B (4 bits com três sementes); AWQ do Llama e do Qwen3-4B (4 bits): em execução. DWQ válido no 0,6B e 1,7B.
- Hipóteses (vereditos **provisórios** do `analise.py`, 25/10 ~17h, sem gravar `hipoteses.json`): **H1** aguardando o AWQ do Llama e do Qwen3-4B (hoje 2 modelos avaliáveis); **H2** inconclusiva (ordem 0,6B < 1,7B < 4B confirmada em 1 de 2 células do RTN); **H3** inconclusiva sem o SmolLM3 (incrementos médios de L 0,39 → 0,51) e **refutada** com ele (0,46 → 0,42) — as duas versões serão reportadas; **H4** aguardando a fase de velocidade. Análise secundária da H3 (log): superlinear nos 5 modelos no RTN (média de S 9,4; IC 9,1–9,6).

## 6. Pendências

1. ~~AWQ 3, 2 e 8 bits do Llama e do Qwen3-4B~~ — concluídos em 26/10 (Fase 16); sobram GPTQ 8 e 2 bits do Llama e do Qwen3-4B e o DWQ do SmolLM3 (8 jobs).
2. DWQ: repetir o diagnóstico até salvar o modelo, com o cache do MLX limitado; comparar a qualidade com o DWQ padrão.
3. ~~Fase de velocidade (H4)~~ — concluída em 26/10 (Fases 17–18): H4 corroborada (provisória); resta, se quiser, **repetir com o app Claude fechado** para eliminar a ressalva de contaminação (20 min).
4. Repetir o tempo do `Qwen3-0,6B-fp` (8 h 53) para confirmar a hipótese do swap (script não agendado).
5. ~~Desligar o agendamento `launchd` `com.edgeai.etapa4`~~ — desligado em 26/10 (Fase 23).
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
