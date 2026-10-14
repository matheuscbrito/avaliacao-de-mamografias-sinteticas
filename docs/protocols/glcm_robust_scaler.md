# Decisão metodológica: Robust Scaler para GLCM

Percentis globais de 1% a 99% produziram clipping excessivo em alguns cortes. Pixels escuros passaram a se acumular no tom 0, aumentando artificialmente pares iguais e, portanto, a energia da GLCM. Isso é artefato de discretização, não evidência de textura repetitiva sintética.

O protocolo adotado usa cercas externas de Tukey com `k = 3,0`, calibradas exclusivamente nos pixels de ROI das CTs reais. O `k = 1,5` chegou a saturar até 19,7% dos pixels no tom 31; com `k = 3,0`, a saturação máxima no teto caiu para 5,1%, preservando melhor a informação vascular.

Para cada par, original e sintética passam pelo mesmo clipping e pelos mesmos 32 tons. Assim, a comparação mede diferença de textura em vez de diferença criada pela régua. Sob essa configuração, 36 de 37 pares Dsr ficaram abaixo da identidade no contraste GLCM, compatível com suavização e perda de altas frequências.
