# Avaliação de CT pulmonar sintética

Este repositório compara CTs pulmonares reais com gerações sintéticas Dgen e Dsr. A organização separa o trabalho em duas frentes:

- **Avaliador de textura**: reunirá GLCM, Wavelet e NEXP em uma integração futura.
- **Filtro de brilho excessivo**: controle de qualidade para identificar pulmões artificialmente claros antes da análise de textura.

## Estrutura atual

- `frameworks/texture_evaluator/glcm/`: módulo GLCM ativo, com Robust Scaler, régua compartilhada de 32 tons e execução por manifesto.
- `wavelet/`: implementação Wavelet já existente; foi preservada sem alteração e será integrada ao avaliador principal em etapa posterior.
- `docs/protocols/`: decisões metodológicas oficiais.
- `docs/contracts/`: contrato de integração para GLCM, Wavelet, NEXP e filtro de brilho.
- `resultados/comparacao_Dgen_Dsr_wavelet/`: resultados Wavelet atuais.
- `archive/`: código, resultados, figuras, slides e documentos históricos, separados por grupo.
- `dados/`: conjunto local original; requer tratamento apropriado antes de qualquer publicação.

## Protocolo GLCM vigente

Para 37 pares válidos, a análise usa a ROI dos dois pulmões erodida com raio 2. A régua de 32 tons é calibrada somente com pixels das imagens reais, por cercas externas de Tukey (`k = 3,0`), e reutilizada sem mudança para a imagem sintética do mesmo par. Contraste e energia são calculados somente entre pixels vizinhos que permanecem dentro da ROI erodida.

Detalhes de execução e interpretação estão em [`frameworks/texture_evaluator/glcm/README.md`](frameworks/texture_evaluator/glcm/README.md).

## Nota sobre interpretação

Resultados de textura e ROC/AUC mostram o quanto imagens reais e sintéticas são distinguíveis. Uma AUC alta detecta diferença entre os grupos; não é, isoladamente, validação de qualidade clínica da síntese.
