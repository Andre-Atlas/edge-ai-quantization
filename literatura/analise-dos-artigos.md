# Análise dos artigos: pontos relevantes, diferenças e uso no artigo

> Baseada em resumos e conhecimento prévio; **nenhum artigo foi lido na íntegra**. Confirmar nas fichas de leitura (`metodologia/ficha-modelo.md`).

## Surveys (referencial teórico)
| Trabalho | Ponto central | Uso no artigo |
|---|---|---|
| Menghani 2023 (ACM CSUR) | Taxonomia de *Efficient DL*: compressão, aprendizado eficiente, automação, arquiteturas, infraestrutura | Esqueleto da seção de referencial; situa a quantização no todo |
| Shuvo 2023 (Proc. IEEE) | Inferência em dispositivos restritos, co-projeto HW/SW e aceleradores (mais orientado a hardware) | Embasa as métricas de borda e a escolha do dispositivo |
| Dantas 2024 (Springer) | Revisão geral de poda, quantização, destilação | Contexto e justificativa do recorte (só quantização) |
| Gholami 2022 | Survey específico de quantização (uniforme/não uniforme, simétrica, PTQ/QAT, bit misto) | Vocabulário e definições formais |
| Nagel 2021 | Receitas práticas de PTQ/QAT para implantação | Protocolo experimental (calibração, granularidade) |
| Zhu 2024 (TACL) | Compressão específica para LLMs | Ponte surveys gerais → métodos de LLM |
| Lin 2023 (IEEE CAS) | Panorama TinyML e co-projeto algoritmo–sistema | Contexto do extremo da borda; trabalhos futuros |

## Métodos
- **GPTQ × AWQ:** ambos pós-treino e só de pesos. GPTQ reconstrói os pesos camada a camada usando a Hessiana; AWQ não reconstrói e protege canais salientes por escala guiada por ativações, o que o torna, segundo seus autores, menos sujeito a *overfitting* à calibração. *Uso:* núcleo da comparação (H1, H3).
- **SmoothQuant:** quantiza também ativações (W8A8) migrando outliers para os pesos; depende de kernels INT8. *Uso:* discussão e limitação (não testável nativamente no Mac).
- **QLoRA:** voltado a *treinar* sobre pesos 4-bit. *Uso:* trabalhos futuros.
- **BitNet b1.58:** pesos ternários treinados do zero. *Uso:* limite inferior de bits (H3).
- **MCUNetV2:** o gargalo na borda é o pico de memória de ativações. *Uso:* justifica medir pico de memória e não só tamanho em disco.

## Diferenças
(a) surveys gerais × de quantização × de LLMs × de hardware; (b) pesos-só × pesos+ativações × treinados em baixo bit × ajuste fino; (c) quase todos avaliados em CUDA, e nenhum dos 13 foi mapeado como avaliação em Apple Silicon.

## Algoritmos e decisão
Ver tabela em [`../metodologia/vertente.md`](../metodologia/vertente.md). Usar: RTN, GPTQ, AWQ, DWQ (mlx-lm) e K-quants/imatrix (llama.cpp). Só referencial: SmoothQuant, QLoRA, BitNet, MCUNetV2. DWQ e K-quants **ainda não têm artigo de referência mapeado**.
