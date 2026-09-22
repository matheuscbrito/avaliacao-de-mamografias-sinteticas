# EDA do dataset de CT pulmonar

Este diretório é gerado por `lung_eda.py`. A análise compara cada imagem TIFF
real com sua correspondente sintética. O objetivo é uma triagem inicial da
fidelidade das imagens, antes da validação radiômica de textura.

## Como ler as métricas

| Campo | Em palavras simples | Como interpretar |
| --- | --- | --- |
| `psnr_full` | Mede a força do sinal da imagem em relação ao erro entre real e sintética; é expresso em decibéis (dB). | **Maior é melhor**. Um valor alto indica que, em média, os pixels são mais parecidos. Não garante que a anatomia ou a textura sejam corretas. |
| `ssim_full` | Mede quão semelhante é a estrutura visual: bordas, contraste local e padrões. Vai aproximadamente de 0 a 1. | **Mais perto de 1 é melhor**. É mais alinhado à aparência visual que PSNR, mas ainda não é uma métrica clínica. |
| `mae_full` | Erro absoluto médio: em cada pixel, calcula a diferença entre real e sintética e tira a média. | **Menor é melhor**. Está na mesma escala numérica do TIFF, então só pode ser comparado entre conjuntos pré-processados da mesma forma. |
| `rmse_full` | Parecido com MAE, mas dá peso maior aos erros muito grandes. | **Menor é melhor**. Se RMSE for muito maior que MAE, alguns poucos pontos têm erro especialmente alto. |
| `mask_nonzero_area_fraction` | Fração da imagem onde a máscara fornecida não é zero. | **Não é qualidade.** Serve apenas para entender quanto da imagem foi marcada. A máscara atual pode incluir pulmão, tórax e mesa. |
| `mae_mask_nonzero` | MAE calculado apenas dentro da parte não-zero da máscara fornecida. | **Menor é melhor**, mas ainda não é MAE pulmonar: só será uma medida de pulmão após validar ou gerar uma máscara pulmonar confiável. |
| `original_mean_mask_nonzero` / `generated_mean_mask_nonzero` | Intensidade média na região marcada, na imagem real e sintética. | Ajuda a detectar mudança global de brilho/escala dentro da região marcada. |
| `original_std_mask_nonzero` / `generated_std_mask_nonzero` | Variação das intensidades na região marcada. | Uma queda forte na sintética pode indicar suavização; aumento pode indicar ruído/heterogeneidade. É um indício, não uma conclusão de textura. |
| `mask_labels` | Valores que aparecem na máscara daquele caso. | Registrar isto é importante porque o dataset usa valores como 0, 0,5 e 1; o significado anatômico ainda precisa ser confirmado. |

## Limites desta EDA

As imagens são TIFFs float, sem metadados DICOM nem unidades Hounsfield (HU).
Por isso, os números descrevem fidelidade na escala disponibilizada pelo
dataset, e não uma validação física de CT. Os PNGs são apenas prévias visuais;
as métricas usam os TIFFs.

## Próxima etapa recomendada

1. Confirmar com quem gerou os dados o significado dos valores 0,5 e 1 da
   máscara fornecida.
2. Gerar/revisar uma máscara pulmonar com `lungmask` em uma amostra.
3. Manter apenas cortes que realmente contenham pulmão e erodir levemente sua
   borda.
4. Extrair GLCM, GLRLM, GLSZM e NGTDM dentro dessa ROI pulmonar, usando os
   mesmos parâmetros para cada par real/sintético.
