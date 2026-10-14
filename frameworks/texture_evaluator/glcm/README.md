# GLCM — protocolo oficial de textura

Este módulo é a parte implementada do avaliador principal. Ele calcula **contraste** e **energia** da GLCM para CTs reais e sintéticas sob uma única régua de intensidades. Wavelet e NEXP pertencem ao avaliador de textura, mas serão integrados posteriormente.

## Protocolo fixo

1. Validar cada par e obter a máscara pulmonar.
2. Erodir a máscara com raio 2; somente essa ROI entra no cálculo.
3. Reunir os pixels da ROI das **CTs reais válidas** (37 cortes no protocolo atual).
4. Calibrar uma única régua robusta: `Q1 - 3 × IQR` até `Q3 + 3 × IQR`.
5. Aplicar essa mesma régua a cada imagem real e sintética do par; fazer clipping e quantizar em 32 tons, de 0 a 31.
6. Montar GLCMs simétricas nas direções 0°, 45°, 90° e 135°. Somente pares de pixels vizinhos que permanecem dentro da ROI erodida são válidos.
7. Extrair a média direcional de contraste e energia, mantendo também a saturação nos tons 0 e 31 como controle técnico.

No protocolo oficial atual, a calibração real foi: `Q1 = -0,163`, `Q3 = 0,226`, `IQR = 0,389`, piso `-1,33`, teto `1,39`, `k = 3,0` e 32 tons. Esses números são registro da coorte atual; uma nova coorte deve recalibrar a régua apenas com seus reais.

## Uso no código

```python
from src import calibrate_robust_scale, extract_glcm

# pooled_real_roi_pixels: pixels reais de todas as ROIs erodidas aceitas
scale = calibrate_robust_scale(pooled_real_roi_pixels, tukey_k=3.0, levels=32)

# A mesma scale deve ser reutilizada para os dois membros do par.
real_features = extract_glcm(real_image, eroded_roi, scale)
synthetic_features = extract_glcm(synthetic_image, eroded_roi, scale)
```

## Execução em lote

Crie um CSV de manifesto; caminhos relativos são interpretados a partir da pasta do próprio CSV.

```csv
case_id,real_path,synthetic_path,mask_path
case_001,data/real/case_001.tiff,data/synthetic/case_001.tiff,data/masks/case_001.tiff
```

Então, a partir de `frameworks/texture_evaluator/glcm/`, execute:

```bash
python -m src.run_glcm --manifest /caminho/manifest.csv --output-dir /caminho/outputs/glcm
```

O processo valida dimensões, erode a máscara com raio 2, calibra a régua uma única vez com todos os pixels reais aceitos e salva `glcm_features.csv`, `glcm_calibration.json` e `rejected_cases.csv`.

O módulo não seleciona casos, não gera máscaras e não decide aprovação. Essas responsabilidades ficam na orquestração futura, para que GLCM, Wavelet e NEXP recebam a mesma lista de pares e ROI.

## Saída mínima por imagem

- `glcm_contrast`: mudança média de tom entre vizinhos; menor valor sugere perda de microtextura/suavização.
- `glcm_energy`: concentração da distribuição de pares de tons.
- `saturation_tone_0_pct` e `saturation_tone_31_pct`: controle contra clipping excessivo.

Não comparar valores absolutos de contraste entre janelas de intensidade diferentes: o contraste depende aproximadamente de `1 / W²`, em que `W` é a largura da janela. Comparações só são válidas dentro da mesma régua compartilhada.
