# Avaliação de textura em CT pulmonar sintética

Este repositório compara duas gerações sintéticas para os mesmos cortes de CT
pulmonar: **Dgen** e **Dsr**. A análise usa GLCM e wavelet dentro da região do
pulmão.

## Organização

- `dados/Dgen/`: conjunto anterior, com original, sintética e máscara.
- `dados/Dsr/`: nova geração; contém os mesmos originais e suas
  sintéticas, mas não traz máscaras próprias.
- `glcm/`: código da análise GLCM e dashboard.
- `wavelet/`: código da análise wavelet e dashboard.
- `resultados/`: tabelas da comparação entre as duas gerações.
- `slides/`: apresentação atual e imagens para inserir em novos slides.
- `documentacao/`: descrição do framework.

## Comparação Dgen × Dsr

Rode a partir de `glcm/`:

```bash
../.venv/bin/python -m src.compare_generations \
  --dgen-dir ../dados/Dgen \
  --dsr-dir ../dados/Dsr \
  --output-dir ../resultados/comparacao_Dgen_Dsr
```

O script só reutiliza as máscaras Dgen após confirmar que os 100 cortes
originais são idênticos em Dgen e Dsr. Ele seleciona os 37 cortes com ROI
pulmonar válida pela regra operacional do MVP e grava as razões de exclusão
dos demais 63. As montagens visuais aprovadas ficam em `slides/imagens/`.

## Resultado inicial

No protocolo GLCM de 32 tons e intervalo global calculado nas originais:

- Dsr teve menor erro absoluto de intensidade na ROI em 35 dos 37 pares;
- a mediana do MAE caiu de 0,378 (Dgen) para 0,151 (Dsr);
- visualmente, Dsr preserva vasos e variações do parênquima que Dgen tende a
  apagar.

Isso não equivale a uma validação clínica. Dsr ainda tem contraste GLCM abaixo
do original na mediana e deve ser avaliado também com wavelet e as demais
métricas antes de uma decisão de uso.

## Resultado wavelet

A mesma comparação com features wavelet está em
`resultados/comparacao_Dgen_Dsr_wavelet/resumo.md`. O Dsr ficou mais próximo
da original em energia (26 de 37 pares) e em entropia (31 de 37) e acompanha a
textura de cada corte (Spearman ≈ 0,8). Já a energia do Dgen tem correlação
negativa com a original. As duas gerações perdem em torno de 40% da energia de
alta frequência.
