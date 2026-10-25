# Edge AI & Quantization

Repositório de pesquisa para um artigo científico sobre **Edge AI e quantização**: estudo experimental comparativo de quantização pós-treino (RTN, GPTQ, AWQ, DWQ e K-quants) de pequenos LLMs (0,6–4B) em um dispositivo de borda real, um MacBook Pro Apple M5 Pro com 24 GB de memória unificada. Eixos: qualidade × memória × velocidade, de 8 a 2 bits. Hipóteses falseáveis H1–H4 com critérios fixados antes dos resultados.

## Onde está cada coisa

| Para saber… | Leia |
|---|---|
| O que cada etapa constitui, o que foi feito e o que falta | [`metodologia/processo.md`](metodologia/processo.md) |
| As regras de decisão (pré-registro) | [`metodologia/protocolo.md`](metodologia/protocolo.md) (v1, tag `protocolo-v1`) e [`metodologia/protocolo-v2.md`](metodologia/protocolo-v2.md) (vigente, tag `protocolo-v2`) |
| Cada passo da execução da matriz, com o raciocínio, as decisões e os erros | [`metodologia/caderno-de-laboratorio.md`](metodologia/caderno-de-laboratorio.md) |
| A lógica do código de quantização, as alterações e as validações | [`experimentos/LOGICA-QUANTIZACAO.md`](experimentos/LOGICA-QUANTIZACAO.md) |
| Como rodar o pipeline e o incidente de memória de 22/10 | [`experimentos/README.md`](experimentos/README.md) |
| A vertente, a lacuna e as hipóteses em linguagem simples | [`metodologia/vertente.md`](metodologia/vertente.md) |

## Estado (25/10/2026)

| Etapa | Estado |
|---|---|
| 0. Ambiente e piloto | ✅ |
| 1. Protocolo | ✅ v1 e v2 |
| 2. Pipeline robusto | ✅ (ampliado na etapa 4) |
| 3. llama.cpp (K-quants) | 🟡 qualidade dos 5 modelos feita; velocidade pendente |
| 4. Matriz completa | 🟡 51/72 jobs de qualidade válidos; AWQ dos modelos grandes em execução; velocidade pendente |
| 5. Análise das hipóteses | 🟡 código pronto; vereditos finais dependem da etapa 4 |
| 6. Painel (GitHub Pages) | 🟡 publicado com o piloto; atualizar ao fechar a etapa 4 |
| 7. Literatura e redação | 🟡 rodada 2 feita; fichas e redação pendentes |

Achado principal até aqui: o GPTQ do mlx-lm 0.32.2 produz modelos inválidos (perplexidade NaN) em 3 dos 5 modelos, por arredondamento em bf16 da Hessiana; detalhes em `protocolo-v2.md`, seção 4.

## Estrutura

```
literatura/
  mapeamento.md            # trabalhos por categoria, com status de verificação (rodadas 1 e 2)
  referencias.bib          # BibTeX
  matriz-de-sintese.csv    # matriz de síntese (ano, veículo, base, contribuição)
  analise-dos-artigos.md   # pontos relevantes, diferenças e uso no artigo
  fichas/                  # fichas de leitura (pendentes)
metodologia/
  processo.md              # etapas: o que constitui, o que foi feito, o que falta
  protocolo.md             # protocolo v1 (congelado)
  protocolo-v2.md          # protocolo v2 (vigente): mudanças desde a v1
  caderno-de-laboratorio.md# cronologia da etapa 4, raciocínio, decisões e incidentes
  vertente.md, popper.md, severino.md, estrutura-do-artigo.md, ficha-modelo.md
experimentos/
  rodar.py, avaliar.py, matriz.py, matriz.json, analise.py, kquant.py, exportar_site.py, testes.py
  gptq_camada.py, awq_liberar.py, dwq_sublote.py      # implementações de menor memória
  testar_hessiana.py, medir_pico_quant.py, perfil_awq.py, comparar_gptq.py, analise_h3_nll.py
  *.sh                     # orquestração e vigilância da execução
  LOGICA-QUANTIZACAO.md    # documentação técnica
  resultados/              # resultados.csv, nll/, kquant.csv, pico-*.csv, h3_nll.*, backups/
site/                      # painel (GitHub Pages)
artigo/                    # rascunhos do manuscrito (vazio por enquanto)
```

Fora do controle de versão (ver `.gitignore`): modelos quantizados (`experimentos/modelos/`), llama.cpp e GGUFs (`experimentos/ferramentas/`), logs (`experimentos/logs/`) e o ambiente (`.venv/`).

## Fundamentação metodológica
- **Karl Popper**, *A lógica da pesquisa científica*: hipóteses falseáveis, testes severos, corroboração provisória.
- **Antônio Joaquim Severino**, *Metodologia do Trabalho Científico*: estrutura principal do projeto e do texto (leitura em etapas, fichamento, projeto, artigo, ABNT).

Os arquivos em `metodologia/` sobre os dois autores são fichamentos em palavras próprias, escritos sem acesso aos livros: devem ser conferidos com a sua edição antes de citar.
