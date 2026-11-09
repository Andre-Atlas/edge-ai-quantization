# Ficha de leitura — #20 Egiazarian et al. (2025)

**Referência (ABNT):** EGIAZARIAN, Vage et al. Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization. **arXiv preprint arXiv:2509.23202**, 2025. Disponível em: https://arxiv.org/abs/2509.23202. Acesso em: 22 out. 2026.
**Chave BibTeX:** `egiazarian2025bridging`
**Data da leitura:** 2026-11-09
**Escopo da leitura:** Consolidação dos metadados e resumo verificados via API do arXiv (`2509.23202` v3, veículo de publicação a conferir), sem fabricação de números de página ou citações literais entre aspas de PDF não lido na íntegra.

## 1. Análise textual (visão global)
- **Tipo de texto:** Preprint algorítmico e experimental (grupo IST Austria / Neural Magic / Red Hat, mesmos coautores do `GPTQ` original: Ashkboos, Hoefler, Alistarh) sobre quantização pós-treino em formatos de ponto flutuante de 4 bits com microescala (`MXFP4` e `NVFP4`).
- **Estrutura conceitual:** Análise das limitações numéricas dos formatos padronizados `MXFP4` (escala compartilhada em potência de 2 do tipo `E8M0` a cada grupo de 32 elementos) e `NVFP4` (escala `FP8` a cada 16 elementos), demonstração de que o arredondamento direto (`RTN`) ou o `GPTQ` padrão degradam na presença de *outliers*, e proposição do algoritmo `MR-GPTQ` (que combina rotações de Hadamard em blocos com reconstrução de segunda ordem estilo GPTQ).
- **Vocabulário-chave:** *microscaling formats*, *MXFP4*, *NVFP4*, *MR-GPTQ*, *block Hadamard rotation*, *power-of-two scale (E8M0)*, *post-training quantization*.

## 2. Análise temática
- **Tema:** Superação da perda de qualidade na quantização pós-treino (PTQ) de LLMs para os novos formatos de hardware de 4 bits baseados em microescala (`MXFP4` e `NVFP4`).
- **Problema:** Embora GPUs modernas de datacenter e consumo (como NVIDIA Blackwell e AMD CDNA) incorporem suporte nativo de hardware aos formatos `MXFP4` e `NVFP4`, a quantização direta para `MXFP4` sofre perda severa de acurácia porque a escala de grupo (`E8M0`) é restrita a potências exatas de 2 e a malha `FP4` (`E2M1`) possui apenas 16 níveis representáveis.
- **Tese / hipótese do(s) autor(es):** A combinação de transformações ortogonais de Hadamard por bloco (que suavizam *outliers* dentro de cada microbloco de 16 ou 32 pesos) com a compensação de erro de segunda ordem do `GPTQ` (`MR-GPTQ`) recupera grande parte da lacuna de qualidade do `MXFP4`/`NVFP4` sem inviabilizar a aceleração de hardware.
- **Argumentação / método:** Os autores analisam teoricamente e empiricamente o erro de quantização do `MXFP4` e do `NVFP4` frente à quantização inteira (`INT4`) e avaliam o `MR-GPTQ` em famílias modernas de LLMs.
- **Resultados principais:** O estudo demonstra que: (i) sem compensação algorítmica, `MXFP4` fica aquém do esperado pela restrição da escala `E8M0`; (ii) `NVFP4` (grupo 16 e escala `FP8`) preserva muito mais qualidade que `MXFP4` (grupo 32 e escala `E8M0`); e (iii) o `MR-GPTQ` reduz substancialmente a degradação em ambos os formatos.

## 3. Análise interpretativa
- **Contexto e relevância:** Escrito pela equipe que criou o `GPTQ` (#08), este trabalho representa a fronteira de 2025–2026 em PTQ de 4 bits, mostrando que mesmo em 4 bits a estrutura da escala, o tamanho do grupo e a precisão da reconstrução de segunda ordem continuam sendo determinantes críticos.
- **Pressupostos:** Assume hardware com suporte nativo (ou projetado para suportar) multiplicação matricial em `MXFP4`/`NVFP4` (arquiteturas NVIDIA Blackwell / AMD).
- **Pontos fortes:** Esclarece com rigor matemático por que a representação das escalas (`bfloat16` afim vs. `FP8` vs. `E8M0` potência de 2) altera drasticamente a retenção de qualidade em 4 bits.
- **Limites:** Preprint sem veículo revisado por pares confirmado; os formatos `MXFP4` e `NVFP4` não possuem aceleração nativa por hardware de tensor cores em Apple Silicon (onde o `MLX` opera com quantização afim inteira `INT2`/`INT3`/`INT4`/`INT8` com escala e viés em `bfloat16`/`float16` por grupo de 64 pesos).

## 4. Problematização
Questões e conexões diretas que o texto suscita para o nosso estudo experimental no Apple M5 Pro (24 GiB):
- **Por que `INT4` afim (grupo 64) no `MLX` atinge alta retenção na H1:** Egiazarian et al. (2025) mostram que a principal fraqueza do `MXFP4` é a escala restrita a potência de 2 (`E8M0`) sem ponto zero flexível. Em contraste, no nosso estudo no Apple M5 Pro, o `mlx-lm` utiliza quantização afim inteira com escala *e* viés em ponto flutuante (`bf16`) a cada 64 pesos ($4{,}5\text{ bits}$ efetivos por peso), permitindo que o `AWQ` 4 bits alcance $Q = 0{,}9702$ no `Qwen3-1.7B` e $Q = 0{,}9554$ no `Llama-3.2-3B`, e o `GPTQ-fp32` 4 bits alcance $Q = 0{,}9287$ no `Qwen3-0.6B`.
- **Sensibilidade da reconstrução estilo GPTQ (conexão com a nossa falha `bf16` vs. `fp32`):** O fato de o próprio grupo autor do `GPTQ` desenvolver o `MR-GPTQ` para estabilizar a reconstrução de segunda ordem frente a *outliers* dialoga diretamente com a nossa descoberta central na Etapa 4/4b: na ausência de precisão suficiente na acumulação da Hessiana ($H = X X^\top$ em `bf16` no `mlx-lm`), o amortecimento diagonal ($\lambda = 0{,}01 \cdot \bar{h}_{ii}$) é arredondado para zero pelo espaçamento de máquina do `bfloat16` ($\text{ulp} \approx 2^{-7}$), levando à falha da decomposição de Cholesky nos modelos de 3B e 4B — totalmente sanada pela nossa acumulação em `float32` (`gptq_fp32`).
- **Limitação explícita e trabalhos futuros:** Justifica no nosso manuscrito por que `MXFP4`/`NVFP4` entram na discussão de formatos modernos de 4 bits, mas não são testados nativamente em Metal no M5 Pro.

## 5. Síntese pessoal
Egiazarian et al. (2025) demonstram que a eficácia da quantização de 4 bits depende criticamente da expressividade das escalas de grupo e da estabilidade da atualização de segunda ordem do `GPTQ`. Para o nosso artigo, o trabalho fundamenta a discussão sobre formatos de 4 bits (contrastando `MXFP4`/`NVFP4` em GPUs discretas com o formato afim `g64` do `MLX` e os superblocos `Q4_K_M` do `llama.cpp`) e reforça a importância de condicionamento numérico adequado ao aplicar `GPTQ`.

## 6. Falseabilidade (Popper)
- **A hipótese central do trabalho é falseável?** Sim. As hipóteses de que (i) a escala `E8M0` do `MXFP4` degrada significativamente a qualidade frente ao `INT4`/`NVFP4` sob `RTN` e (ii) o `MR-GPTQ` reduz essa perda são diretamente refutáveis por medições de perplexidade e acurácia zero-shot.
- **Como foi testada?** Avaliando `RTN`, `GPTQ` e `MR-GPTQ` nos formatos `MXFP4` e `NVFP4` em benchmarks padronizados de LLMs.
- **Quais condições limitam a conclusão?** Os ganhos de velocidade reais do `MR-GPTQ` exigem suporte de hardware para `MXFP4`/`NVFP4` e kernels eficientes para aplicar as rotações de Hadamard em blocos em tempo de inferência.

## Citações úteis / Pontos-chave verificados para citação indireta
- **Seções de uso previstas no artigo:** Referencial Teórico (`R`), Discussão (`D`) e Trabalhos Futuros (`F`).
- **Ponto verificado 1 (Limites dos formatos microscaling FP4):** Os formatos emergentes `MXFP4` e `NVFP4` sofrem perda não trivial de qualidade em PTQ direta devido às restrições de representação das escalas compartilhadas (especialmente a escala em potência de 2 do `MXFP4`), exigindo técnicas como rotações de Hadamard combinadas ao `GPTQ` (`MR-GPTQ`) (EGIAZARIAN et al., 2025).
- **Ponto verificado 2 (Contraste com o formato afim do MLX no Apple Silicon):** Enquanto `MXFP4` e `NVFP4` dependem de aceleradores específicos de GPUs NVIDIA/AMD, o ecossistema Apple Silicon (`MLX`) emprega quantização inteira afim com escalas e vieses em `bfloat16` por grupo de 64 pesos, evitando a restrição de escala em potência de 2 (EGIAZARIAN et al., 2025).
