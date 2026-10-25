# Lógica da quantização e das ferramentas de apoio — documentação técnica

> Documento de rastreabilidade do estudo. Descreve **o que o código faz**, **o que foi alterado ou criado em relação ao pipeline de 22/10/2026**, **por quê**, **como cada alteração foi validada** e **o que não está validado**. A cronologia dos passos está em `metodologia/caderno-de-laboratorio.md`; as regras de decisão do estudo, em `metodologia/protocolo-v2.md` (vigente, tag `protocolo-v2`) e `metodologia/protocolo.md` (v1, congelado na tag `protocolo-v1`); o estado de cada etapa, em `metodologia/processo.md`.
> Redigido em 25/10/2026, com o estado do código às ~17h. Os números vêm de `resultados/*.csv` e `logs/*.log`.

## 0. Ambiente e versões

| Item | Valor |
|---|---|
| Máquina | Apple M5 Pro, 24 GiB de memória unificada (`hw.memsize` = 25 769 803 776 bytes = 24,00 GiB = **25,77 GB decimais**) |
| Sistema | macOS 27.0.1 |
| Python / MLX / mlx-lm | 3.14.9 / 0.32.4 / **0.32.2** (`uv.lock`) |
| huggingface_hub / numpy | 2.2.0 / 2.5.4 |
| llama.cpp (K-quants) | v0.7.0 (620d222), binários em `experimentos/ferramentas/llama.cpp/build/bin` |
| Commit base do repositório | `acf5b9d`; tag congelada `protocolo-v1` |

**Convenção de unidades.** Todas as memórias neste estudo estão em **GB decimais (10⁹ bytes)**, que é o que `psutil`, `mx.get_peak_memory()` e o `time -l` devolvem divididos por 10⁹. Os "24 GB" da máquina são 24 GiB = 25,8 GB decimais. Tamanhos de modelo (`fp_gb`) e picos usam a mesma escala.

**Duas medidas de memória, que não são a mesma coisa.**
- *Pico do MLX* (`mx.get_peak_memory()`): máximo da memória **ativa** alocada pelo MLX (Metal). Não conta buffers liberados que ficaram no cache do MLX nem memória fora do MLX.
- *Footprint do processo* (`peak memory footprint` do `/usr/bin/time -l`): memória física do processo inteira, incluindo cache do MLX, Python, mapeamentos. É o que pressiona o sistema. Foi **0,6 a 2,3 GB maior** que o pico do MLX nos jobs medidos.

## 1. Pipeline original (22/10/2026), em uma página

1. `matriz.py` lê `matriz.json` (modelos × métodos × bits × sementes), gera a lista de *jobs*, aplica o guarda de memória e chama `rodar.py` por job.
2. `rodar.py` (um job = um modelo × método × largura):
   1. `quantizar()` gera o modelo em `experimentos/modelos/<ident>`: `mlx_lm convert` (fp e RTN), `mlx_lm gptq|awq|dwq` (métodos com calibração). Grava `tamanho_gb` e `tempo_quant_s` no CSV (status `quantizado`).
   2. `avaliar.py --fase qualidade`: perplexidade no WikiText-2 (janelas de 2048 tokens, uma por vez, NLL por janela salva em `resultados/nll/<ident>.npy`) e acurácia zero-shot em 5 tarefas (lm-eval). Grava no CSV (status `qualidade`).
   3. `avaliar.py --fase velocidade` (fase separada, máquina ociosa).
3. `analise.py` aplica os critérios do protocolo (H1–H4) com intervalos de confiança por *bootstrap* pareado sobre as janelas.
4. Chave de linha no CSV: `modelo, metodo, bits, grupo, seed`. `gravar()` atualiza a linha com a mesma chave sob trava de arquivo (`fcntl`).
5. `qualidade` Q = PPL_fp / PPL_q.

## 2. Alterações em arquivos já existentes

### 2.1 `experimentos/rodar.py` (+46 / −1 linhas; `git diff experimentos/rodar.py`)

| # | Alteração | Lógica | Motivo |
|---|---|---|---|
| R1 | `quantizar()` passa a executar o comando com `/usr/bin/time -l` via `Popen`, repassando o *stderr* ao log em tempo real e extraindo `peak memory footprint` | Imprime `[pico_quant] <metodo> <modelo> <bits>b: footprint X GB` no log do job; levanta `CalledProcessError` se o código de saída ≠ 0 | Medir o pico **real** de cada quantização em todo job futuro (validar a regra de memória fora da amostra) |
| R2 | Opção `--gptq-camada` (ou `EAQ_GPTQ_CAMADA=1`) | Para `gptq`, troca `python -m mlx_lm gptq …` por `gptq_camada.py … --hessiana bf16` | Quando a implementação padrão não cabe na RAM (§3.1) |
| R3 | Opção `--awq-liberar` (ou `EAQ_AWQ_LIBERAR=1`) | Para `awq`, troca `python -m mlx_lm awq …` por `awq_liberar.py …` | Remover a retenção de memória do AWQ padrão (§3.3) |
| R4 | Coluna extra `impl` no CSV | Preenchida com `gptq_camada.py (Hessiana bf16, identico ao mlx-lm)` ou `awq_liberar.py (gc por bloco, identico ao mlx-lm)` quando a implementação alternativa é usada | Rastrear qual implementação gerou cada linha. **Linhas anteriores a 25/10 não têm `impl`: usaram o `mlx_lm` padrão** |
| R5 | `import re, os` | — | Suporte às alterações acima |

### 2.2 `experimentos/matriz.py` (+86 / −4 linhas; `git diff experimentos/matriz.py`)

| # | Alteração | Lógica | Motivo |
|---|---|---|---|
| M1 | `excluido(cfg, j)` e filtro em `jobs()` | Remove da lista os jobs cuja combinação `(modelo, metodo, bits)` casa com uma entrada de `excluir` em `matriz.json`; campos ausentes casam com qualquer valor | Falhas de ferramenta (GPTQ 3 bits, AWQ no SmolLM3) e limites de memória; evita repetir jobs que falham (o GPTQ 3 bits do SmolLM3 gastou 43 min até falhar) |
| M2 | `arq` repassado de `matriz.json` para cada job | Dimensões da arquitetura (§4.1) | Calcular a memória das Hessianas |
| M3 | Guarda de memória recalibrado (`estimativa_gb`, `necessario_sozinho_gb`, `pico_fp_rtn_gb`, `hessianas_gb`, `pico_gptq_awq_gb`, `usa_camada`, `pico_gptq_camada_gb`, `memoria_total_gb`) | Fórmulas na §4.2 | A estimativa antiga (`fp_gb × 1,5 + 4`) estava **alta demais para fp/rtn** (a fila ficou ~4,5 h parada esperando 14,1 GB livres num Mac com 12,6 GB, sendo que o pico medido do maior job era 10,5 GB) e **baixa demais para GPTQ e AWQ** (14,7 e 24,2 GB medidos contra 10,1 GB estimados no Qwen3-1,7B) |
| M4 | `comando()` acrescenta `--gptq-camada` quando `usa_camada(j)` e `--awq-liberar` para todo AWQ | A matriz escolhe sozinha a implementação alternativa | Sem isso, remover a exclusão do GPTQ do Qwen3-4B faria a matriz rodar a implementação padrão (estimativa de 57 GB) |

### 2.3 `experimentos/matriz.json`

- **`arq`** por modelo: `d` (hidden), `i` (intermediária), `n` (camadas), `o_in` (entrada do `o_proj` = cabeças × dim. da cabeça). Lidos dos `config.json` do Hugging Face.
- **`excluir`** (estado em 25/10/2026, 17h): GPTQ 3 bits (todos os modelos); AWQ no SmolLM3; DWQ no Qwen3-4B; DWQ no Llama-3.2-3B. Histórico: entradas de GPTQ e AWQ do Qwen3-4B e AWQ do Llama foram **criadas em 25/10 e removidas** depois que o GPTQ camada a camada e o AWQ com liberação por bloco ficaram validados. Cópias anteriores em `resultados/backups/`.
- Cada entrada traz o `motivo` com a evidência.

### 2.4 `experimentos/analise.py` (+98 / −43 linhas; protocolo v2, seção 5)

- Cada dado tem **três estados**: presente e válido; **pendente** (ausente e fora de `excluir`: ainda pode chegar); permanentemente ausente (em `excluir`). Linhas com `status` iniciado por `falhou` não entram (inclui as invalidadas por NaN).
- `pendente(L, m, metodo, bits)`; `SEPARADOS = [SmolLM3-3B]` e `CONTADOS`: o SmolLM3 sai das contagens de H1 e H3 (caso à parte, decisão de 25/10).
- **H1:** "algum método ciente de dados" = os disponíveis em 4 bits por modelo (antes exigia os três). `rtn_ok` conta sempre; `nenhum` só é definitivo quando todos os métodos possíveis estão presentes. Veredito antecipado: `refutada` se `rtn_ok ≥ 3` ou `nenhum ≥ 3`; `corroborada` se `sucesso ≥ 4`; `aguardando dados` só se há dado pendente que poderia mudar o veredito; senão `inconclusiva`. **Limiares absolutos da v1 mantidos.**
- **H2:** células = RTN × {4, 3 bits} nos três Qwen3; GPTQ/AWQ/DWQ só em 0,6B e 1,7B ficam como suplementar.
- **H3:** 1ª metade (incrementos de L = 1 − Q do RTN, modelos com RTN completo); 2ª metade (melhor entre GPTQ e DWQ disponíveis em 3 bits contra o RTN, ≥ 20% de redução); contagem de "sem redução" ≥ 3 refuta cedo.
- **Sensibilidade:** `main()` calcula H1 e H3 também com o SmolLM3 contado e imprime/grava (`sensibilidade_com_caso_separado`).
- Saída JSON rotulada `protocolo-v2`.
- Original preservado em `resultados/backups/analise.py.protocolo-v1` (e no git, `HEAD`).

## 3. Ferramentas novas e a lógica de cada uma

**Critério de equivalência usado em todas as validações:** tensores idênticos bit a bit **e** NLL por janela idêntica contra a implementação padrão, em pelo menos dois modelos. É mais forte que comparar a perplexidade: com tensores idênticos, qualquer métrica derivada também é. Quando a alternativa **não** é idêntica (GPTQ em float32, DWQ com alvos truncados), usa-se a razão Q com IC 95% pareado e a regra escrita antes do teste (protocolo-v2, seção 7). O raciocínio por trás de cada escolha (e as alternativas descartadas) está em `metodologia/caderno-de-laboratorio.md`, seção 1A.

### 3.1 `gptq_camada.py` — GPTQ camada a camada

**Problema.** O `mlx_lm.quant.gptq` instala um acumulador de Hessiana (`Catcher`, `H += xᵀx`) em **todas** as camadas lineares, roda o modelo e só então quantiza. As Hessianas de todas as camadas ficam vivas ao mesmo tempo (acumuladores float32 + produtos bf16 do lote ≈ 1,5× o tamanho delas): 14,7 GB no Qwen3-1,7B.

**Algoritmo.**
```
semente = S; carregar modelo (preguiçoso); carregar calibração (todas as amostras, 512 tokens; lotes de 8)
acts = embedding(lotes)                         # ativações do modelo ORIGINAL
para cada camada i:
    envolver as nn.Linear da camada em Captura (acumula H = Σ xᵀx, igual ao Catcher)
    para cada lote h em acts: novas.append(camada(h))     # forward com pesos em precisão cheia
    para cada Linear: Hinv = inversa_da_hessiana(H); quantizar_linear(...)   # laço copiado do mlx-lm
    trocar as Linear pelas quantizadas; liberar H
    acts = novas                                 # a próxima camada recebe ativações do modelo ORIGINAL
(lm_head amarrado: não há) ; embedding → a 6 bits (reserva); salvar como o mlx-lm
```
**Por que é o mesmo método.** No `mlx_lm`, todas as Hessianas vêm de **uma única passada no modelo original**; aqui cada camada recebe as ativações do modelo original (não as do quantizado). Os acumuladores, o amortecimento (1% da média da diagonal), a fatoração, o laço de atualização de pesos e o empacotamento (`quantize` importado do `mlx_lm`) são os mesmos; a semente é fixada na mesma ordem (semente → carga do modelo → dados), então as amostras e os lotes coincidem.

**Modos.** `--hessiana bf16` (padrão do mlx-lm; produto `xᵀx` em bf16 somado em float32) e `--hessiana fp32` (produto em float32; **desvio**, §3.2).

**Validação (Qwen3, 4 bits, grupo 64, semente 0, `bf16`) contra `mlx_lm gptq`:**

| | 0,6B | 1,7B |
|---|---|---|
| Tensores idênticos | 704 / 704 | 704 / 704 |
| `config.quantization` | idêntica | — |
| max\|ΔNLL por janela\| (146 janelas) | 0 | 0 |
| PPL | 22,9027 = 22,9027 | 18,1849 = 18,1849 |
| Pico do MLX (padrão → camada a camada) | 5,8 → 1,44 GB | 14,7 → 3,31 GB |
| Footprint (camada a camada) | 2,54 GB | 6,87 GB |
| Tempo | 112 → 123 s | 310 → 466 s (466 s com concorrência de outro job) |

**Modo `fp32` não reproduz o padrão:** no 0,6B, PPL 22,58 contra 22,90; razão Q_novo/Q_padrão = 1,0143 (IC 95% pareado 1,0127–1,0159, **não contém 1**). Portanto o GPTQ em float32 é **diagnóstico**, não substitui o padrão (protocolo-v2, seção 7, A).

**Limites.** Não suporta janela deslizante (asserção). Equivalência verificada em dois Qwen3; nos demais modelos o modo `bf16` reproduz também as **falhas** do padrão (NaN). Footprint no Qwen3-4B: 17,9 GB (previsto 17,5).

### 3.2 `testar_hessiana.py` — por que o GPTQ dá NaN

**Hipótese (originada da leitura do `gptq.py`):** o `Catcher` acumula `xᵀx` em bf16; em matrizes quase singulares o arredondamento pode torná-las indefinidas, o Cholesky devolve NaN e o NaN se propaga às colunas seguintes.

**Método.** Percorre o modelo camada a camada (CPU opcional, `--cpu`), coleta a Hessiana de `q_proj` (= k, v), `o_proj`, `gate_proj` (= up) e `down_proj` em duas versões — `H_orig` (como o mlx-lm: produto bf16) e `H_exata` (float32) —, repete `compute_inverse_hessian` (amortecimento de 1%, Cholesky, inversa, Cholesky superior) e conta NaN. Para matrizes ≤ 4096 calcula o menor autovalor (`eigvalsh`).

**Resultados (todas as camadas de cada modelo).**

| Modelo | Matrizes que falham com `H_orig` (bf16) | Com `H_exata` (fp32) |
|---|---|---|
| SmolLM3-3B | `o_proj` 0, 5, 34; `down_proj` 10 | nenhuma |
| Llama-3.2-3B | `o_proj` 0 | nenhuma |
| Qwen3-4B | `o_proj` 0, 1, 24, 26, 30; `down_proj` 6, 35 | nenhuma |

Exemplos de menor autovalor da `H_orig`: Llama `o_proj` 0 = −2,42 (amortecimento 0,41); Qwen3-4B `o_proj` 30 = −403 (254); SmolLM3 `o_proj` 34 = −103 (25,5). **Não é mau condicionamento** (razão diagonal/amortecimento de 10³–10⁴ em todas as camadas, inclusive nas saudáveis). Não foi investigado por que só essas matrizes são quase singulares.

### 3.3 `perfil_awq.py` e `awq_liberar.py` — AWQ sem retenção de memória

**Problema.** AWQ padrão: 13,4 GB (0,6B) e 24,2 GB (1,7B) de pico do MLX.

**Diagnóstico (`perfil_awq.py`).** Envolve `run_layer`, `search_best_scale`, `search_best_clip`, `apply_scale`, `scale_block` e `clip_block` e imprime memória ativa, pico acumulado e cache a cada chamada, **sem alterar nenhum cálculo**. No 0,6B: a memória ativa ao fim de cada bloco **cresce ~1,4 GB por bloco** e despenca no fim. O tamanho das entradas guardadas por bloco (`input_feat`: 65 536 tokens × 2 B × (3·1024 + 2048 + 2·1024 + 3072) = 1,34 GB) coincide com esse crescimento. Leitura do `awq.py`: `Catcher` atribui `module.input_feat` e nunca o apaga.

**Correção (`awq_liberar.py`).** (a) `mx.clear_cache()` — que o `awq.py` já chama ao fim de cada bloco — passa a ser precedido de `gc.collect()`; (b) `mx.set_cache_limit(1 GB)` (o cache chegou a 11,5 GB e conta no footprint). Nenhuma delas altera valores.

**Mecanismo — o que está provado e o que não está.** Provado empiricamente: com a correção a memória ativa fica **plana** entre blocos (2,70 → 2,94 GB em 28 blocos, +0,01 GB/bloco) e o resultado é bit a bit idêntico. **Não** foi identificado qual referência exata mantinha as entradas vivas; a hipótese é coleta de lixo cíclica atrasada de grafos de módulos (o perfil original mostra quedas periódicas de memória, típicas de coletas).

**Validação contra `mlx_lm awq` (4 bits, grupo 64, semente 0):**

| | 0,6B | 1,7B |
|---|---|---|
| Tensores idênticos | 704 / 704 | 704 / 704 |
| max\|ΔNLL por janela\| | 0 | 0 |
| PPL | 23,7196 = 23,7196 | 17,2334 = 17,2334 |
| Pico do MLX (padrão → liberado) | 13,42 → 4,17 GB | 24,2 → 8,44 GB |
| Footprint (liberado) | 5,43 GB | 9,30 GB |
| Tempo | 112 → 109 s | 445 → 325 s |

**Ruído entre sementes do AWQ (4 bits):** PPL 23,72 / 23,62 / 23,96 (0,6B; desvio 0,18 = 0,8%) e 17,23 / 17,12 / 17,34 (1,7B; 0,11 = 0,7%).

### 3.4 `dwq_sublote.py` — DWQ com alvos do professor em sub-lotes

**Problema.** No DWQ, o professor em precisão cheia e o aluno ficam na memória, com logits completos por passo. `--target-dir` pré-calcula os alvos do professor e dispensa o professor no treino, **mas** os alvos guardam só os **1024 maiores logits** (`argpartition`) — a perda é truncada, então o resultado **não é idêntico** ao DWQ padrão (logits completos) — e o cálculo dos alvos consome memória ∝ lote × tokens × vocabulário (12,4 GB de MLX e 14–20 GB de footprint já no 0,6B).

**Lógica.** Substitui `compute_dwq_targets` por uma versão que processa **uma amostra por vez**, concatena os resultados e grava **um arquivo por lote** (a indexação que o treino exige). Linhas são independentes ⇒ mesmos alvos a menos de ruído numérico.

**Verificação (CPU, lote aleatório 4 × 256, Qwen3-0,6B):** 100% dos índices iguais (como conjuntos por posição) e diferença máxima 0 nos logits ordenados. **Não verificado na GPU** (lotes e linhas podem diferir por arredondamento de bf16).

**Resultado do diagnóstico real (0,6B, 4 bits, máquina livre, 25/10 16:18):** fase de alvos concluída (6,6 GB em disco); treino chegou a ~400 dos 512 passos (perda estável, ~2 850 tokens/s) com pico do MLX de **5,89 GB**, mas **footprint de 20,05 GB** — a trava de segurança (1,5 GB livres) abortou antes de salvar. **Sem comparação de qualidade ainda.** O hiato 5,9 → 20 GB sugere cache do MLX e retenção do Python (hipótese não testada).

### 3.5 `medir_pico_quant.py` — medição de pico de memória

Roda uma quantização num processo próprio, no mesmo comando do `rodar.quantizar` (`runpy` do módulo `mlx_lm`), com: pico do MLX (`mx.get_peak_memory`), amostragem a cada 0,5 s da memória disponível do sistema e do swap (`psutil`), **trava de segurança** (encerra o processo se a memória disponível cair abaixo de `--piso-gb`) e footprint via `/usr/bin/time -l`. Saídas: `resultados/pico-quantizacao.csv` (MLX) e `resultados/pico-footprint.csv` (footprint). Opções `--saida` (mantém o modelo) e `--extra` (argumentos extras da ferramenta, p.ex. `--target-dir`; o caminho **não pode ter espaços**).

**Medições (4 bits, grupo 64):**

| Modelo | Método | Pico MLX | Footprint | Observação |
|---|---|---|---|---|
| Qwen3-0,6B | GPTQ | 5,80 | 8,12 | 112 s |
| Qwen3-0,6B | AWQ | 13,42 | 15,58 | swap +1,9 GB |
| Qwen3-0,6B | DWQ | > 6,14 | — | abortado pela trava (3 GB) |
| Qwen3-1,7B | GPTQ | 14,66 | — | swap +3,8 GB; 310 s |
| Qwen3-1,7B | AWQ | 24,20 | — | swap +7,6 GB; 445 s |
| Qwen3-1,7B | DWQ | > 9,56 | — | abortado pela trava (1,5 GB) |
| Llama-3.2-3B | GPTQ (padrão) | — | 26,06 | 2 522 s |
| Qwen3-4B | GPTQ (camada a camada) | — | 17,90 | 930 s |
| 0,6B / 1,7B | fp (convert) | 1,19 / 3,44 | 1,83 / 4,07 | conversão |
| 0,6B / 1,7B | rtn (convert) | 1,53 / 2,27 | 2,20 / 3,83 | conversão |
| Qwen3-0,6B | DWQ `--target-dir` (fase de alvos) | 12,42 | 14,13 → 19,83 | abortado em 12 e 16 s (a 2ª, sozinha) |

### 3.6 `comparar_gptq.py` — comparação de modelos quantizados

Compara dois diretórios de modelo tensor a tensor (pesos empacotados `uint32`, escalas e *biases*) e, opcionalmente, a NLL por janela do WikiText-2 do modelo novo contra o `.npy` do padrão, com a razão Q_novo/Q_padrão e IC 95% por *bootstrap* pareado (10 000 reamostragens). Usado em todas as validações de equivalência acima.

### 3.7 `analise_h3_nll.py` — análise secundária (pós-hoc) da H3

`D_b = média das janelas de (NLL_q,b − NLL_fp) = ln(PPL_q / PPL_fp)` em nats por token; `S = (D₂ − D₃) − (D₃ − D₄)`; superlinear se o IC 95% de S > 0. Motivo: Q satura em 0 aos 2 bits, e o incremento de 3→2 bits de L = 1 − Q tem **teto** igual ao Q de 3 bits. Também reproduz o critério original e a 2ª metade da H3. Não substitui o veredito original. Resultado parcial em `resultados/h3_nll.md`.

## 4. Fórmulas do guarda de memória (estado em 25/10/2026)

### 4.1 Dimensões (`matriz.json`)

| Modelo | `fp_gb` | d | i | n | o_in |
|---|---|---|---|---|---|
| Qwen3-0,6B | 1,2 | 1024 | 3072 | 28 | 2048 |
| Qwen3-1,7B | 4,1 | 2048 | 6144 | 28 | 2048 |
| SmolLM3-3B | 6,2 | 2048 | 11008 | 36 | 2048 |
| Llama-3.2-3B | 6,4 | 3072 | 8192 | 28 | 3072 |
| Qwen3-4B | 8,1 | 2560 | 9728 | 36 | 4096 |

### 4.2 Fórmulas

- **fp / rtn:** `pico = max(fp_gb + 0,5; tamanho + 2,7)`, com `tamanho = fp_gb` (fp) ou `fp_gb × (bits + 0,5)/16` (rtn). Base: 24 linhas medidas — pico da perplexidade = tamanho do modelo + 1,9 a 2,7 GB; conversão ≤ tamanho bf16 (0,6B e 1,7B). `estimativa = pico + 1,5`; `inicia com pico + 2`.
- **GPTQ padrão:** `pesos + 1,5 × H + 1,5 GB` (MLX), com `H = 4 × n × (5d² + o_in² + i²)/10⁹`; footprint = MLX × 1,4. Ajustado em 0,6B e 1,7B. **Extrapolação a modelos grandes superestima** (Llama: previsto 40,2 GB, medido 26,06 GB).
- **GPTQ camada a camada** (`usa_camada`: pico padrão > RAM): `2,1 × fp_gb + 0,5` (previsto 17,5 GB no Qwen3-4B; medido 17,90).
- **AWQ** (sempre `awq_liberar`): `4,0 + 1,4 × fp_gb` (cobre 5,43 e 9,30 GB medidos). Extrapolação (Llama ≈ 13,0; Qwen3-4B ≈ 15,3 GB) **não validada**.
- **DWQ: não calibrado** (só limites inferiores): mantém `fp_gb × 3 + 4`.
- **Limiar de partida** (GPTQ/AWQ): `min(estimativa + 2; 0,5 × RAM total)` (12,9 GB). Jobs maiores que a RAM rodam sozinhos com swap (SmolLM3 GPTQ, AWQ 1,7B), sem travar.

## 5. Scripts de automação e vigilância

| Script | Função |
|---|---|
| `executar_etapa4.sh` (existente) | Pipeline completo (qualidade → K-quants → velocidade → análise), retomável |
| `agendar_llama_gptq.sh`, `agendar_qwen4b_gptq.sh`, `agendar_awq_grandes.sh` | Rodam um job específico, em série e sozinhos; ao fim conferem a perplexidade (`[conferencia]`: detectam NaN) |
| `continuar_lista.sh` | Réplicas de AWQ e K-quants; **não** inicia a velocidade |
| `testar_dwq_alvos.sh`, `rodar_diagnosticos.sh` | Testes/diagnósticos DWQ (alvos) e perfil do AWQ, sozinhos |
| `vigiar_espera.sh` | Sai com `STALL:` se a fila ficar > 1 h esperando memória (ou viva, sem job e sem escrita no log); o assistente é acordado e envia e-mail |
| `repetir_fp_0.6B.sh` | Repetição do tempo do Qwen3-0,6B fp (8 h 53 min na 1ª rodada); **não está agendado hoje** |

## 6. Integridade dos dados

- **Linhas inválidas** (status `falhou:invalido-nan`, coluna `obs`, métricas originais preservadas, ignoradas por `analise.py`): GPTQ 4 bits do Llama-3.2-3B e do Qwen3-4B; GPTQ 8/4/2 bits do SmolLM3. Todas com PPL = NaN e acurácia média 0,3438 (nível do acaso).
- **Linhas de falha por ferramenta** (`falhou:qualidade`): GPTQ 3 bits (todos), AWQ do SmolLM3. As do Llama (erro 401, modelo restrito antes da aprovação da Meta) foram sobrescritas ao rodar.
- **Backups** com checksum SHA-256 em `resultados/backups/`.
- O `resultados.csv`, as NLL por janela (`resultados/nll/`), as medições de memória e os backups entram no git a partir de 25/10/2026; as logs (`experimentos/logs/`) e os modelos continuam fora (`.gitignore`).

## 7. Pendências, riscos e o que não está validado

1. DWQ (alvos em sub-lotes): treino e salvamento não concluídos; sem comparação de qualidade contra o padrão.
2. Extrapolações de memória a 3B e 4B (AWQ liberado, GPTQ camada a camada de outros modelos): a validar com os `[pico_quant]` dos jobs em andamento (AWQ do Llama e do Qwen3-4B).
3. Equivalência de `gptq_camada.py` e `awq_liberar.py` com o `mlx_lm` demonstrada em Qwen3-0,6B e 1,7B; nos modelos Llama e Qwen3-4B deve ser conferida (re-executar a comparação com a implementação padrão onde couber).
4. Origem exata da retenção de memória do AWQ não identificada.
5. Por que só poucas matrizes são quase singulares no GPTQ: não investigado.
6. A fase de velocidade (H4) não foi iniciada; K-quants concluídos para os cinco modelos.
7. ~~`protocolo-v2` rascunho; `protocolo.md` modificado~~ — resolvido em 25/10: v2 vigente (tag `protocolo-v2`), `protocolo.md` idêntico à tag `protocolo-v1`, `analise.py` aplicando a v2.

## 8. Checksums SHA-256 do código e do protocolo no commit de 25/10/2026 (12 primeiros caracteres)

Calculados imediatamente antes dos commits de 25/10/2026; os documentos (caderno, processo, READMEs, este arquivo) não entram porque mudam a cada atualização — para eles vale o histórico do git. Compare com `shasum -a 256 <arquivo>`.

| Arquivo | Linhas | SHA-256 |
|---|---|---|
| `experimentos/gptq_camada.py` | 203 | `fa692a912ff8` |
| `experimentos/awq_liberar.py` | 68 | `83ef7dccd472` |
| `experimentos/dwq_sublote.py` | 116 | `d6084a3aecb6` |
| `experimentos/testar_hessiana.py` | 147 | `4cccee44647c` |
| `experimentos/medir_pico_quant.py` | 134 | `652ad7440d90` |
| `experimentos/perfil_awq.py` | 71 | `44c325ae0176` |
| `experimentos/comparar_gptq.py` | 62 | `01d1c97ae066` |
| `experimentos/analise_h3_nll.py` | 147 | `54bd8aa132f5` |
| `experimentos/analise.py` | 224 | `2430b3c82212` |
| `experimentos/matriz.py` | 217 | `2027cef07d83` |
| `experimentos/rodar.py` | 208 | `0d17f54ca69f` |
| `experimentos/matriz.json` | 98 | `b87866b9f463` |
| `experimentos/vigiar_espera.sh` | 41 | `e63eacee9573` |
| `experimentos/agendar_llama_gptq.sh` | 20 | `c260655f9555` |
| `experimentos/agendar_qwen4b_gptq.sh` | 29 | `135d56597bd6` |
| `experimentos/agendar_awq_grandes.sh` | 29 | `6eec9c2ba26a` |
| `experimentos/testar_dwq_alvos.sh` | 28 | `0cafb6222018` |
| `experimentos/continuar_lista.sh` | 22 | `fd4ac6263d7c` |
| `experimentos/rodar_diagnosticos.sh` | 40 | `b471a0aca0e7` |
| `experimentos/repetir_fp_0.6B.sh` | 18 | `241050aa5679` |
| `metodologia/protocolo.md` | 44 | `642834289f36` |
| `metodologia/protocolo-v2.md` | 123 | `df093ffbfa54` |

Backups (`experimentos/resultados/backups/`):

- `analise.py.protocolo-v1` — `7245741450eb`
- `matriz.json.antes-das-dimensoes-de-arquitetura` — `3166ac2897f2`
- `matriz.json.antes-de-reabilitar-awq-llama` — `e41248fe20eb`
- `resultados.csv.antes-de-invalidar-gptq-llama-qwen4b` — `f7a976b37047`
- `resultados.csv.antes-de-invalidar-gptq-smollm3` — `4670be8bab76`
- `resultados.csv.em-25-10-17h` — `7a2c36b5344f`
