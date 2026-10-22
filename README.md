# Edge AI & Quantization

Repositório de planejamento e pesquisa para um artigo científico sobre **Edge AI e Quantização** (inferência eficiente de redes neurais em dispositivos de borda).

## Estrutura

```
literatura/
  mapeamento.md            # 13 trabalhos (2021–2024), por categoria, com status de verificação
  referencias.bib          # BibTeX
  matriz-de-sintese.csv    # matriz de síntese (ano, veículo, base, contribuição)
  analise-dos-artigos.md   # pontos relevantes, diferenças e uso no artigo
metodologia/
  popper.md                # falseabilidade, corroboração e aplicação ao projeto
  severino.md              # leitura, fichamento, projeto e estrutura do artigo
  estrutura-do-artigo.md   # esqueleto do artigo e perguntas em aberto
  vertente.md              # vertente, hipóteses H1–H4, limitações
  ficha-modelo.md          # modelo de ficha de leitura
experimentos/             # pipeline MLX: rodar.py, avaliar.py, resultados/ (ver experimentos/README.md)
artigo/                    # rascunhos do manuscrito (vazio por enquanto)
```

## Fundamentação metodológica
- **Karl Popper**, *A lógica da pesquisa científica*: hipóteses falseáveis, testes severos, corroboração provisória.
- **Antônio Joaquim Severino**, *Metodologia do Trabalho Científico*: estrutura principal do projeto e do texto (leitura em etapas, fichamento, projeto, artigo, ABNT).

Os arquivos em `metodologia/` são fichamentos em palavras próprias, escritos sem acesso aos livros: devem ser conferidos com a sua edição antes de citar.

## Próximos passos
1. Decidir tipo de artigo, foco e veículo (ver `metodologia/estrutura-do-artigo.md`).
2. Ler os 13 trabalhos e preencher fichas (`literatura/fichas/`).
3. Segunda rodada de busca (2025–2026, QAT, NPUs, benchmarks de borda).
4. Formular a hipótese falseável e o critério de refutação.
