# Fundamentação: Karl Popper — *A lógica da pesquisa científica*

> Fichamento de apoio **em palavras próprias**, feito de memória sobre a obra (*Logik der Forschung*, 1934; ed. inglesa 1959). Não substitui a leitura do livro: ao citar, confira a edição e as páginas que você usa e preencha as lacunas marcadas com `[pág.]`.

## Ideias centrais

1. **Problema da indução e dedutivismo.** Nenhum número finito de observações confirma logicamente uma teoria universal. A ciência procede por *conjecturas* testadas dedutivamente.
2. **Demarcação.** O critério que separa ciência empírica de não-ciência é a **falseabilidade**: uma teoria é científica se proíbe certos eventos observáveis e, portanto, pode ser refutada pela experiência.
3. **Teste por dedução.** Derivam-se consequências da teoria e confrontam-se com resultados experimentais; uma discordância reprodutível refuta a teoria (ou exige revisar hipóteses auxiliares).
4. **Corroboração (não confirmação).** Uma teoria que resiste a testes severos está *corroborada* provisoriamente, nunca provada.
5. **Conteúdo empírico e risco.** Quanto mais uma hipótese proíbe, mais informativa e mais arriscada; hipóteses *ad hoc* que blindam a teoria contra a refutação reduzem seu valor científico.
6. **Base empírica e reprodutibilidade.** Enunciados básicos são aceitos por convenção e só valem se o efeito for reprodutível e intersubjetivamente testável.

## Como isso se traduz para o artigo (Edge AI & Quantization)

| Princípio de Popper | Aplicação no projeto |
|---|---|
| Hipóteses falseáveis | Formular afirmações do tipo: *"A quantização PTQ de 4 bits mantém ≥ X% da acurácia do modelo FP16 em tarefa T, com redução ≥ Y× de memória e latência ≤ Z ms no dispositivo D"*, que um experimento pode refutar. |
| Testes severos | Escolher cenários onde a hipótese tem chance real de falhar: bit-widths baixos, modelos pequenos (mais sensíveis), outliers de ativação, hardware real e não só simulação. |
| Evitar manobras *ad hoc* | Fixar de antemão métricas, *datasets*, conjunto de calibração e critério de refutação; não ajustar a posteriori o limiar de "aceitável". |
| Reprodutibilidade | Publicar código, *seeds*, versões de bibliotecas e especificações do hardware. |
| Corroboração provisória | Redigir conclusões como "resultados corroboram a hipótese nas condições testadas", sem generalizar além delas. |

## Pendências de leitura
- [ ] Capítulos sobre falseabilidade e grau de falseabilidade `[pág.]`
- [ ] Capítulo sobre corroboração `[pág.]`
- [ ] Capítulo sobre base empírica `[pág.]`
