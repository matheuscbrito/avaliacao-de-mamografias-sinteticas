# Comparação Dgen × Dsr — wavelet

Mesmo protocolo da comparação GLCM (`../comparacao_Dgen_Dsr`): 100 pares com
originais idênticas, máscaras do Dgen reutilizadas e os mesmos 37 cortes válidos.
Features pelo método de imagem inteira do MVP wavelet (haar, 2 níveis, fração
mínima de tecido 0,9, intervalo de clipping calculado nas originais).

| Métrica | Dgen | Dsr |
| --- | --- | --- |
| Pares em que a energia ficou mais próxima da original | 11 | **26** |
| Pares em que a entropia ficou mais próxima da original | 6 | **31** |
| Mediana do erro de entropia, em IQRs da referência | 0,69 | **0,35** |
| Mediana do erro de energia, em IQRs da referência | 1,01 | 0,97 |
| Correlação de Spearman com a original: energia | **−0,43** | **0,77** |
| Correlação de Spearman com a original: entropia | 0,24 | **0,79** |
| Energia sintética / original (mediana) | 0,59 | 0,63 |
| Classificação do framework: approved / manual_review | 8 / 29 | 9 / 28 |

Diferenças pareadas nos erros (Wilcoxon): energia p = 0,0004; entropia
p = 0,00001.

## Leitura

- **Dsr segue a textura de cada corte; Dgen não.** A energia do Dgen tem
  correlação negativa com a original: cortes com mais detalhe fino saem mais
  lisos, e cortes lisos ganham textura. No Dsr a ordenação dos cortes se
  mantém (ρ ≈ 0,8), o que confirma a leitura visual e o resultado de MAE da
  análise GLCM.
- **As duas gerações perdem energia de alta frequência.** Ambas guardam em
  torno de 60% da energia de detalhe original, em todas as sub-bandas. O Dsr
  perde de forma mais uniforme (caixas estreitas em
  `figuras/energia_por_subbanda.png`). Na sub-banda diagonal mais fina (HH e1),
  ele perde um pouco mais que o Dgen, o que é compatível com a remoção de ruído.
  Isso concorda com o contraste GLCM abaixo da original no Dsr.
- **A classificação do framework quase não muda (8 → 9 aprovadas).** Ela é
  dominada pelo erro de energia acima de 1 IQR, que é sistemático nas duas
  gerações. Como o Dsr erra quase sempre na mesma direção, esse viés poderia ser
  medido e discutido. No Dgen, o erro é imprevisível de um corte para outro.

Conclusão: o Dsr é claramente superior ao Dgen na preservação de textura
wavelet, mas ainda suaviza o parênquima. Isso não equivale a uma validação
clínica, e o limiar de 1 IQR continua sendo uma configuração do MVP.

## Reproduzir

A partir de `wavelet/`:

```bash
.venv/bin/python -m src.compare_generations \
  --dgen-dir ../dados/Dgen \
  --dsr-dir ../dados/Dsr \
  --output-dir ../resultados/comparacao_Dgen_Dsr_wavelet
```
