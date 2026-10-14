# Contrato de integração do avaliador de textura

Esta especificação é para a futura junção de GLCM, Wavelet e NEXP. Não implementa as métricas pendentes.

## Entrada comum

Cada execução deve receber um manifesto com `case_id`, caminho da imagem real, caminho da imagem sintética, máscara pulmonar e versão do protocolo. A ROI deve ser erodida uma única vez, com raio 2, antes de ser entregue a qualquer métrica.

## Saída comum

Cada módulo deve devolver uma linha por imagem com `case_id`, `image_type` (`real` ou `synthetic`), nome e versão da métrica, parâmetros e seus valores. A comparação pareada só ocorre depois que as três saídas compartilham o mesmo conjunto de `case_id`.

## Regras

- Não misturar resultados de versões diferentes de pré-processamento.
- GLCM usa a régua robusta calibrada nos reais; Wavelet e NEXP devem declarar explicitamente qualquer normalização própria.
- ROC/AUC avalia capacidade de detectar diferenças entre reais e sintéticas. AUC alta é alerta de separação, não selo de qualidade da síntese.
