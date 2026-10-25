# Protocolo experimental — versão 2

> **Status: vigente desde 25/10/2026 (tag `protocolo-v2`).** Não substitui `protocolo.md` (congelado na tag `protocolo-v1`, que permanece inalterado): registra, com data e motivo, o que mudou depois daquela tag, como exige o aviso da v1. Qualquer mudança de **regra** a partir daqui exige uma `protocolo-v3`. O **estado de execução** da grade (o que já rodou) não é regra e é acompanhado em `metodologia/caderno-de-laboratorio.md`.

## 1. O que muda e o que não muda

**Não muda:** as definições operacionais (Q, Q_acc, E, intervalos), as quatro hipóteses H1–H4 e **todos os limiares** (95% de retenção, 75% das células, 20% de redução, E ≥ 0,6 e E < 0,4). O veredito principal de cada hipótese continua sendo o da v1.

**Muda:** (a) a grade realmente executada, por limites da ferramenta e da memória; (b) duas implementações de menor memória que reproduzem o método padrão bit a bit; (c) como as regras de contagem são aplicadas quando faltam dados; (d) uma análise secundária da H3, declarada pós-hoc; (e) o registro de uma falha de método e das análises complementares.

## 2. Transparência: o que já tinha sido visto antes desta versão

- O piloto (Qwen3-0,6B, 4 bits), como já declarado na v1.
- Entre 23 e 25/10/2026, os resultados parciais da matriz: Qwen3-0,6B e 1,7B (todos os métodos), SmolLM3-3B (RTN; GPTQ inválido), Llama-3.2-3B e Qwen3-4B (RTN completo; GPTQ inválido).
- A análise secundária da H3 (seção 6) foi concebida **depois** de ver que o critério original falha por teto em três modelos; por isso é pós-hoc.
- **Antes da tag**, também foram vistos os **vereditos provisórios** do `analise.py` (25/10, ~17h): H2 inconclusiva; H3 inconclusiva sem o SmolLM3 e **refutada** com ele; H1 e H4 aguardando dados. **Nenhuma regra foi ajustada depois disso**: as regras da seção 5 e a decisão de tratar o SmolLM3 como caso à parte são anteriores a esses vereditos (caderno, Fase 7).
- Nenhum limiar foi alterado depois de ver resultados.

## 3. Grade executada: lacunas e implementações

Ferramenta: mlx-lm 0.32.2. Hardware: Apple M5 Pro, 24 GiB (25,8 GB decimais) de memória unificada. Implementação das lacunas: chave `excluir` em `experimentos/matriz.json`.

| Combinação | Situação | Motivo |
|---|---|---|
| GPTQ a 3 bits, todos os modelos | não existe | `AssertionError: Unsupported bits 3` (o GPTQ do mlx-lm só aceita 2, 4 e 8 bits) |
| AWQ no SmolLM3-3B | não existe | `NotImplementedError: AWQ support for smollm3 models NYI` |
| GPTQ no Qwen3-4B | executado com `gptq_camada.py` (3.1) | a implementação padrão não cabe na RAM |
| AWQ no Llama-3.2-3B e no Qwen3-4B | executado com `awq_liberar.py` (3.2); 4 bits em execução em 25/10 | a implementação padrão não cabe na RAM |
| DWQ no Llama-3.2-3B e no Qwen3-4B | excluído | memória: no Qwen3-1,7B o DWQ foi abortado pela trava de segurança; a variante de menor memória ainda não fecha (7, D) |
| DWQ no SmolLM3-3B | não executado; caso à parte (seção 4) | memória, por extrapolação |

**Medições de memória** (`experimentos/medir_pico_quant.py`; `resultados/pico-quantizacao.csv` e `pico-footprint.csv`), 4 bits: GPTQ padrão 5,8 GB (0,6B) e 14,7 GB (1,7B) de pico do MLX; AWQ padrão 13,4 e 24,2 GB; DWQ abortado nos dois (> 6,1 e > 9,6 GB). O *footprint* do processo foi 0,6–2,3 GB maior que o pico do MLX. A estimativa original do `matriz.py` (10,1 GB para GPTQ e AWQ no 1,7B) estava baixa. Detalhes e fórmulas em `experimentos/LOGICA-QUANTIZACAO.md`.

### 3.1 GPTQ camada a camada (`experimentos/gptq_camada.py`)

O `mlx_lm gptq` mantém as Hessianas de **todas** as camadas ao mesmo tempo (≈ 1,5× o tamanho delas). O `gptq_camada.py` percorre o modelo camada a camada (coleta a Hessiana, quantiza, solta), usando como entrada de cada camada as ativações do modelo **original**, como no `mlx_lm`; o laço de quantização, a semente, a calibração (calibration_v5, todas as amostras, 512 tokens, lote de 8) e as camadas de reserva (embedding a 6 bits) são os mesmos. Com `--hessiana bf16` (o padrão do mlx-lm):

| Qwen3, 4 bits, semente 0 | 0,6B | 1,7B |
|---|---|---|
| Tensores idênticos ao `mlx_lm gptq` | 704 de 704 | 704 de 704 |
| Diferença máxima da NLL por janela | 0 | 0 |
| Pico do MLX (padrão → camada a camada) | 5,8 → 1,4 GB | 14,7 → 3,3 GB |

**É o mesmo método, não um desvio.** Usado onde o padrão não cabe (`matriz.py`: `usa_camada`; flag `--gptq-camada`; coluna `impl` no CSV). Reproduz também as falhas do padrão (NaN, seção 4).

### 3.2 AWQ com liberação de memória por bloco (`experimentos/awq_liberar.py`)

O perfil de memória (`perfil_awq.py`) mostrou que o AWQ padrão acumula ~1,4 GB por bloco no Qwen3-0,6B — o tamanho das entradas guardadas de cada bloco, que o `awq.py` nunca apaga. O `awq_liberar.py` chama `gc.collect()` antes da limpeza de cache que o `awq.py` já faz ao fim de cada bloco e limita o cache do MLX a 1 GB. Nenhum cálculo muda:

| Qwen3, 4 bits, semente 0 | 0,6B | 1,7B |
|---|---|---|
| Tensores idênticos ao `mlx_lm awq` | 704 de 704 | 704 de 704 |
| Diferença máxima da NLL por janela | 0 | 0 |
| Pico do MLX (padrão → liberado) | 13,4 → 4,2 GB | 24,2 → 8,4 GB |

**É o mesmo método, não um desvio.** A matriz o usa para todo AWQ (`--awq-liberar`; coluna `impl`).

## 4. Falha de método: GPTQ padrão do mlx-lm 0.32.2

O GPTQ saiu com perplexidade NaN em todas as janelas e acurácia média 0,3438 (nível do acaso) no **SmolLM3-3B** (8, 4 e 2 bits), no **Llama-3.2-3B** (4 bits, implementação padrão) e no **Qwen3-4B** (4 bits, `gptq_camada.py`, que reproduz o padrão bit a bit; logo a falha é do método, não da reimplementação). Funciona no Qwen3-0,6B e no 1,7B. Linhas marcadas `falhou:invalido-nan`, valores originais preservados, fora da análise.

**Causa demonstrada nos três modelos** (`experimentos/testar_hessiana.py`, todas as camadas): o `Catcher` do `gptq.py` soma produtos `xᵀx` calculados em bf16; em poucas matrizes quase singulares o arredondamento torna a Hessiana indefinida (menor autovalor negativo e maior, em módulo, que o amortecimento de 1% da média da diagonal); o Cholesky devolve NaN e o NaN se propaga às colunas seguintes e ao modelo inteiro.

| Modelo | Matrizes que falham com a Hessiana em bf16 | Em float32 |
|---|---|---|
| SmolLM3-3B | `o_proj` 0, 5, 34; `down_proj` 10 | nenhuma |
| Llama-3.2-3B | `o_proj` 0 | nenhuma |
| Qwen3-4B | `o_proj` 0, 1, 24, 26, 30; `down_proj` 6, 35 | nenhuma |

Não é mau condicionamento (razão diagonal/amortecimento de 10³–10⁴ em todas as camadas). **Não foi investigado** por que só essas matrizes são quase singulares. O GPTQ com Hessiana em float32 não reproduz o padrão (seção 7, A), então não substitui estes resultados.

**SmolLM3-3B como caso à parte (decisão de 25/10/2026).** Sem nenhum método ciente de dados válido (GPTQ inválido, AWQ inexistente, DWQ não executado), fica **fora das contagens de H1 e H3**, que se baseiam em Qwen3-0,6B, Qwen3-1,7B, Llama-3.2-3B e Qwen3-4B. Seus resultados válidos (FP e RTN) são reportados. O `analise.py` calcula, junto de cada veredito, a **sensibilidade com o SmolLM3 contado**, e **as duas versões devem ser reportadas**: no estado de 25/10, a primeira metade da H3 é superlinear sem ele (incrementos médios de L no RTN de 0,39 → 0,51) e não superlinear com ele (0,46 → 0,42), o que leva a H3 de *inconclusiva* para *refutada*.

## 5. Regras de contagem com dados faltantes

Limiares absolutos da v1 mantidos (por exemplo, H1 corroborada com ≥ 4 dos 5 modelos). **Decisão de 25/10/2026: não adotar limiares proporcionais**, mesmo sabendo que isso pode impedir a corroboração de H1 e da segunda metade da H3; "inconclusiva" é a resposta esperada nesses casos, com a evidência reportada modelo a modelo. Modelos sem os dados necessários são **não avaliáveis** e **não contam como sucesso**.

| Hipótese | Regra aplicada | Efeito no estado de 25/10/2026 |
|---|---|---|
| H1 | "Algum método ciente de dados" = os disponíveis em 4 bits por modelo, em vez de exigir os três | Avaliáveis: Qwen3-0,6B e 1,7B; **Llama e Qwen3-4B dependem do AWQ a 4 bits (em execução)**. Com 4 avaliáveis, a corroboração exige os 4; com 2, só pode sair inconclusiva |
| H2 | Células = RTN × {4, 3 bits} nos três Qwen3; GPTQ/AWQ/DWQ (sem o 4B) como comparação par a par, fora da fração | Das 8 células da v1 sobram 2 no denominador; a H2 fica **mais fraca** e deve ser reportada assim |
| H3 | 1ª metade: critério original sobre os modelos com RTN completo. 2ª metade: melhor entre GPTQ e DWQ disponíveis a 3 bits | A 2ª metade só tem Qwen3-0,6B e 1,7B (DWQ): não pode ser corroborada (limiar 4) nem refutada por contagem (limiar 3); a H3 só pode ser refutada pela 1ª metade ou ficar inconclusiva |
| H4 | Inalterada: E com os métodos disponíveis de cada modelo | Nenhum |

O `experimentos/analise.py` aplica esta seção (rótulo `protocolo-v2` no JSON de saída). Estados de um dado: presente; **pendente** (pode chegar); ausente por exclusão. Vereditos monótonos saem cedo; "aguardando dados" só quando um dado pendente poderia mudá-los.

## 6. Análise secundária da H3 em escala logarítmica (pós-hoc)

**Motivo.** Q = PPL_fp/PPL_q satura em 0 aos 2 bits; com L = 1 − Q, o incremento de 3→2 bits tem teto igual ao Q de 3 bits, e o critério original falha por construção quando a perda de 4→3 bits passa de ~0,5. Isso ocorre no Qwen3-0,6B, no 1,7B e no SmolLM3; **não** no Llama-3.2-3B nem no Qwen3-4B, onde o critério original é cumprido (incrementos 0,29 → 0,64 e 0,29 → 0,65).

**Medida.** `D_b = média das janelas de (NLL_q,b − NLL_fp) = ln(PPL_q/PPL_fp)`; `S = (D₂ − D₃) − (D₃ − D₄)`; superlinear se o IC 95% de S > 0 (bootstrap pareado sobre as janelas, 10 000 reamostragens). `experimentos/analise_h3_nll.py` → `resultados/h3_nll.md`/`.json`.

**Papel.** Complementar; **não substitui** o veredito original. Divergências são reportadas e explicadas pelo teto.

**Resultado (25/10/2026, RTN dos cinco modelos).** S > 0 com IC inteiro acima de zero nos cinco: Qwen3-0,6B 8,5; Qwen3-1,7B 8,9; SmolLM3-3B 18,3; Llama-3.2-3B 5,1; Qwen3-4B 6,0; média 9,4 (IC 95% 9,1–9,6). AWQ e DWQ também superlineares nos dois Qwen3 menores. Na 2ª metade, o DWQ reduz L do RTN a 3 bits em 31% (0,6B) e 40% (1,7B).

## 7. Análises complementares

**A. GPTQ com Hessiana em float32 — diagnóstico.** Regra escrita antes do teste: o ajuste deveria reproduzir o GPTQ padrão no Qwen3-0,6B dentro do IC 95%. **Não reproduziu** (PPL 22,58 × 22,90; razão Q 1,0143, IC 1,0127–1,0159). Portanto confirma a causa do NaN, mas não substitui o GPTQ padrão nas contagens.

**B. Estender A aos demais modelos — sem efeito** (A reprovou na verificação). Regra original registrada: só se A fosse considerável (mudar H1/H3 pela sensibilidade, ou Q do SmolLM3 fora do IC dos demais).

**C. Repetição do tempo do Qwen3-0,6B-fp** (8 h 53 contra ~25 min do 1,7B, provavelmente swap). Afeta só o registro de tempo. Pendente.

**D. DWQ com alvos do professor pré-calculados (`dwq_sublote.py`) — variante, não o método padrão.** Os alvos guardam só os 1024 maiores logits (perda truncada). Calculados em sub-lotes (mesmos alvos que em lote, verificado na CPU). No Qwen3-0,6B o treino chegou a ~400/512 passos com 5,9 GB de pico do MLX, mas 20 GB de footprint, e foi abortado pela trava antes de salvar. **Se** vier a fechar, entra como coluna separada e declarada, avaliada pela razão Q com IC contra o DWQ padrão, e só seria aplicada aos modelos grandes se rodada também nos pequenos.

## 8. Ameaças à validade (acréscimos às da v1)

- Grade incompleta por limites da ferramenta e do hardware (24 GiB): as conclusões de H1–H3 valem para o subconjunto executado; a H2 ficou restrita ao RTN no Qwen3-4B.
- `gptq_camada.py` e `awq_liberar.py` têm equivalência bit a bit demonstrada em Qwen3-0,6B e 1,7B; nos demais modelos ela é suposta pela construção (mesmo código de cálculo), não verificada.
- O GPTQ padrão do mlx-lm 0.32.2 falha em 3 dos 5 modelos; versões futuras da ferramenta podem não falhar.
- Uma execução por configuração (semente 0); as réplicas de AWQ indicam ruído de ~0,7–0,8% na PPL a 4 bits.
- A exclusão do SmolLM3 das contagens muda o veredito da H3 (seção 4): as duas versões são reportadas.

## 9. Formalização (25/10/2026)

1. `protocolo.md` restaurado ao conteúdo da tag `protocolo-v1` (as notas acrescentadas em 24–25/10 estão consolidadas aqui).
2. `analise.py` adaptado à seção 5 (rótulo `protocolo-v2`).
3. Commit e tag `protocolo-v2`.

## 10. Rastreabilidade

`metodologia/caderno-de-laboratorio.md` (cronologia, raciocínio de cada ação, decisões, incidentes), `experimentos/LOGICA-QUANTIZACAO.md` (lógica do código, validações, fórmulas, checksums), `metodologia/processo.md` (o que cada etapa constitui, o que foi feito e o que falta) e `experimentos/resultados/backups/` (cópias de segurança).
