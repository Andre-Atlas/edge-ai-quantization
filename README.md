# Edge AI & Quantization

Repositório de pesquisa para um artigo científico sobre **Edge AI e quantização**: estudo experimental comparativo de quantização pós-treino (RTN, GPTQ, AWQ, DWQ e K-quants) de pequenos LLMs (0,6–4B) em um dispositivo de borda real, um MacBook Pro Apple M5 Pro com 24 GB de memória unificada. Eixos: qualidade × memória × velocidade, de 8 a 2 bits. Hipóteses falseáveis H1–H4 com critérios fixados antes dos resultados.

## Onde está cada coisa

| Para saber… | Leia |
|---|---|
| O que cada etapa constitui, o que foi feito e o que falta | [`metodologia/processo.md`](metodologia/processo.md) |
| As regras de decisão (pré-registro) | [`metodologia/protocolo.md`](metodologia/protocolo.md) (v1, tag `protocolo-v1`), [`metodologia/protocolo-v2.md`](metodologia/protocolo-v2.md) (v2, tag `protocolo-v2`) e [`metodologia/protocolo-v3.md`](metodologia/protocolo-v3.md) (v3, tag `protocolo-v3`) |
| Cada passo da execução da matriz, com o raciocínio, as decisões e os erros | [`metodologia/caderno-de-laboratorio.md`](metodologia/caderno-de-laboratorio.md) |
| A lógica do código de quantização, as alterações e as validações | [`experimentos/LOGICA-QUANTIZACAO.md`](experimentos/LOGICA-QUANTIZACAO.md) |
| Como rodar o pipeline e o incidente de memória de 22/10 | [`experimentos/README.md`](experimentos/README.md) |
| A vertente, a lacuna e as hipóteses em linguagem simples | [`metodologia/vertente.md`](metodologia/vertente.md) |
| O manuscrito científico completo, tabelas e figuras | [`artigo/manuscrito.md`](artigo/manuscrito.md) |
| As 27 fichas de leitura estruturadas (Severino + Popper) | [`literatura/fichas/README.md`](literatura/fichas/README.md) |

## Estado (09/11/2026)

| Etapa | Estado |
|---|---|
| 0. Ambiente e piloto | ✅ |
| 1. Protocolo | ✅ v1, v2 e v3 |
| 2. Pipeline robusto | ✅ (`testes.py` 8/8 e `testes_v3.py` 25/25) |
| 3. llama.cpp (K-quants) | ✅ qualidade e velocidade dos 5 modelos |
| 4. Matriz completa | ✅ execução concluída em 26/10 (59 válidos, 9 inválidos por NaN; velocidade medida em 2 rodadas) |
| 4b. Extensão v3 | 🟡 pré-registrada (`protocolo-v3`); validação 704/704 e sanidade 3 bits concluídas no `Qwen3-0,6B`; script `rodar_bloco1_v3.sh` pronto com trava `AC Power` |
| 5. Análise das hipóteses | ✅ fechada para a matriz v2 (`hipoteses.json`): H4 corroborada; H1–H3 inconclusivas; concordância $Q \times Q_{\text{acc}}$ ($r = 0,9520$) e variância $\sigma_{\text{cal}}(Q) = 0,0065$ |
| 6. Painel (GitHub Pages) | ✅ atualizado em 09/11 com a matriz v2, K-quants e validação `gptq_fp32` |
| 7. Literatura e redação | ✅ 27 fichas de leitura (`literatura/fichas/`), 5 tabelas Markdown/LaTeX (`artigo/tabelas/`), 4 figuras SVG (`artigo/figuras/`) e manuscrito completo (`artigo/manuscrito.md`) |

Achados principais:
1. O GPTQ padrão do `mlx-lm 0.32.2` produz modelos inválidos (perplexidade `NaN`) em todos os modelos de 3–4B e larguras testadas, por arredondamento em `bfloat16` da Hessiana ($X^\top X$ perde semidefinição positiva nas camadas finais); corrigido em `gptq_fp32` (`--hessiana fp32`).
2. **H4 corroborada** nos modelos de 3–4B ($E = 0,83\text{--}0,87 \ge 0,60$, aceleração de decodificação de $2,95\times\text{--}3,08\times$ para redução de tamanho de $3,54\times$).
3. Forte concordância entre retenção de perplexidade e retenção de acurácia zero-shot ($r = 0,9520$, $\rho = 0,9413$) e baixo ruído de semente de calibração no AWQ 4 bits ($\sigma_{\text{cal}}(Q) = 0,0065$).

## Estrutura

```
literatura/
  mapeamento.md            # trabalhos por categoria, com status de verificação (rodadas 1 e 2)
  referencias.bib          # BibTeX (27 entradas)
  matriz-de-sintese.csv    # matriz de síntese (ano, veículo, base, contribuição)
  analise-dos-artigos.md   # pontos relevantes, diferenças e uso no artigo
  fichas/                  # 27 fichas de leitura estruturadas (01 a 27) + README.md
metodologia/
  processo.md              # etapas: o que constitui, o que foi feito, o que falta
  protocolo.md             # protocolo v1 (congelado)
  protocolo-v2.md          # protocolo v2 (vigente): mudanças desde a v1
  protocolo-v3.md          # protocolo v3 (extensão pré-registrada)
  caderno-de-laboratorio.md# cronologia das fases 1 a 25, raciocínio, decisões e incidentes
  vertente.md, popper.md, severino.md, estrutura-do-artigo.md, ficha-modelo.md
experimentos/
  rodar.py, avaliar.py, matriz.py, matriz.json, matriz-v3.json, analise.py, analise_v3.py
  kquant.py, exportar_site.py, gerar_tabelas_artigo.py, testes.py, testes_v3.py
  gptq_camada.py, awq_liberar.py, dwq_sublote.py      # implementações de menor memória
  testar_hessiana.py, medir_pico_quant.py, perfil_awq.py, comparar_gptq.py, analise_h3_nll.py
  *.sh                     # orquestração e vigilância da execução (incl. rodar_bloco1_v3.sh)
  LOGICA-QUANTIZACAO.md    # documentação técnica
  resultados/              # resultados.csv, hipoteses.json, nll/, kquant*.csv, pico-*.csv, h3_nll.*, backups/
site/                      # painel interativo (GitHub Pages)
artigo/
  manuscrito.md            # manuscrito científico completo em português (ABNT / Severino + Popper)
  tabelas/                 # 5 tabelas geradas por script (.md e .tex)
  figuras/                 # 4 figuras vetoriais (.svg) geradas por script
```

Fora do controle de versão (ver `.gitignore`): modelos quantizados (`experimentos/modelos/`), llama.cpp e GGUFs (`experimentos/ferramentas/`), logs (`experimentos/logs/`) e o ambiente (`.venv/`).

## Fundamentação metodológica
- **Karl Popper**, *A lógica da pesquisa científica*: hipóteses falseáveis, testes severos, corroboração provisória.
- **Antônio Joaquim Severino**, *Metodologia do Trabalho Científico*: estrutura principal do projeto e do texto (leitura em etapas, fichamento, projeto, artigo, ABNT).

Os arquivos em `metodologia/` sobre os dois autores são fichamentos em palavras próprias, escritos sem acesso aos livros: devem ser conferidos com a sua edição antes de citar.
