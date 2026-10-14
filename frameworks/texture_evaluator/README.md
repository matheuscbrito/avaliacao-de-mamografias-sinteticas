# Avaliador de textura

Este é o espaço do framework principal. Ele deverá reunir três métricas calculadas na mesma coorte, nos mesmos pares e na mesma ROI pulmonar erodida:

- `glcm/`: implementado e documentado neste repositório.
- Wavelet: reservado para integração futura.
- NEXP: reservado para integração futura.

O framework não deve aceitar que cada métrica escolha seus próprios casos, máscara, escala ou pré-processamento. A orquestração futura deve produzir um manifesto de casos válidos e compartilhar as ROIs com os três módulos.
