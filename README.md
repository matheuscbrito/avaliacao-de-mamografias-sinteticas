# Avaliação de textura em CT pulmonar sintética

Este repositório compara duas gerações sintéticas para os mesmos cortes de CT
pulmonar: **Dgen** e **Dsr**. A análise atual usa GLCM dentro da ROI pulmonar
e mantém os materiais de apresentação separados dos resultados reproduzíveis.

## Organização

- `datasets/ct/Dgen/`: conjunto anterior, com original, sintética e máscara.
- `datasets/ct/Dsr/`: nova geração; contém os mesmos originais e suas
  sintéticas, mas não traz máscaras próprias.
- `lung_texture_mvp/`: pipeline e dashboard GLCM.
- `reports/`: CSVs, sínteses e figuras geradas por análises.
- `assets/slide_images/`: imagens prontas para uso em slides.
- `presentations/`: arquivos PowerPoint.
- `docs/`: documentação metodológica.
- `tools/`: scripts auxiliares e análises históricas de Dgen.

## Comparação Dgen × Dsr

Rode a partir de `lung_texture_mvp/`:

```bash
../.venv/bin/python -m src.compare_generations \
  --dgen-dir ../datasets/ct/Dgen \
  --dsr-dir ../datasets/ct/Dsr \
  --output-dir ../reports/dataset_comparison
```

O script só reutiliza as máscaras Dgen após confirmar que os 100 cortes
originais são idênticos em Dgen e Dsr. Ele seleciona os 37 cortes com ROI
pulmonar válida pela regra operacional do MVP e grava as razões de exclusão
dos demais 63. As montagens visuais são copiadas para
`assets/slide_images/dataset_comparison/` quando forem aprovadas para slides.

## Resultado inicial

No protocolo GLCM de 32 tons e intervalo global calculado nas originais:

- Dsr teve menor erro absoluto de intensidade na ROI em 35 dos 37 pares;
- a mediana do MAE caiu de 0,378 (Dgen) para 0,151 (Dsr);
- visualmente, Dsr preserva vasos e variações do parênquima que Dgen tende a
  apagar.

Isso não equivale a uma validação clínica. Dsr ainda tem contraste GLCM abaixo
do original na mediana e deve ser avaliado também com wavelet e as demais
métricas antes de uma decisão de uso.
