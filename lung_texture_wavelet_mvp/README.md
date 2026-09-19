# MVP de validação de textura pulmonar com wavelet

Este projeto avalia se uma CT pulmonar sintética preserva a textura da original que a condicionou e continua plausível perante as CTs reais validadas do mesmo dataset. É o irmão do MVP de GLCM (`../lung_texture_mvp`): mesmo fluxo, mesma leitura de resultados, outra família de features.

Enquanto o GLCM mede relações de coadjacência entre pares de pixels vizinhos, a wavelet decompõe a ROI em sub-bandas de frequência e orientação. As duas abordagens são complementares: a wavelet enxerga textura em múltiplas escalas ao mesmo tempo.

## Fluxo

```text
original + sintética + máscara
              ↓
validação do caso e erosão leve da máscara
              ↓
recorte da ROI pulmonar + decomposição wavelet 2D
              ↓
energia e entropia das sub-bandas de detalhe (LH, HL, HH)
              ↓
referência global feita apenas com originais aceitas
              ↓
comparação sintética × par original + sintética × referência
              ↓
CSV + gráficos + dashboard de leitura
```

## Como rodar

Instale as dependências em um ambiente virtual e execute, a partir desta pasta:

```bash
pip install -r requirements.txt
python -m src.run_pipeline --data-dir /caminho/para/upscale_test_new_lung_Dgen
streamlit run dashboard/app.py
```

O dataset atual é plano: cada original `caso.tiff` precisa ter `caso_generate.tiff` e `caso_mask.tiff`. O MVP também aceita a estrutura `data/original`, `data/synthetic` e `data/masks`.

Parâmetros específicos da wavelet:

- `--wavelet` (padrão `db4`): família wavelet, qualquer nome aceito pelo PyWavelets (`haar`, `db2`, `sym4`, `coif1`...).
- `--level` (padrão `2`): níveis de decomposição. Se a ROI for pequena demais para o nível pedido, o script usa o máximo possível e registra em `levels_used`.

## O que cada script faz

- `src/load_data.py`: encontra os três arquivos de cada caso e abre as imagens.
- `src/validate_cases.py`: aceita somente pares que correspondem e máscaras com dois pulmões utilizáveis; grava os motivos de descarte.
- `src/wavelet_features.py`: recorta a ROI pulmonar, decompõe com DWT 2D e mede energia e entropia por sub-banda.
- `src/build_reference.py`: cria a distribuição de referência das originais aceitas.
- `src/compare_images.py`: compara a sintética tanto com a original do par quanto com a referência.
- `src/run_pipeline.py`: orquestra tudo e salva CSVs e figuras.
- `dashboard/app.py`: lê os arquivos prontos; não recalcula features.

## Como a ROI entra na wavelet

A DWT opera sobre a imagem inteira, não sobre pares isolados de pixels como o GLCM. Por isso o MVP recorta a caixa delimitadora da máscara erodida e preenche o que está fora do pulmão com a mediana da própria ROI. Sem esse preenchimento, a borda nítida entre pulmão e fundo viraria o padrão de alta frequência dominante e mascararia a textura que queremos medir.

## As duas features

- **Energia** (`energy`): média dos coeficientes de detalhe ao quadrado. Cresce quando a ROI tem mais estrutura fina — bordas, reticulação, ruído de alta frequência.
- **Entropia** (`entropy`): dispersão dos coeficientes de detalhe, em bits. Cresce quando a energia se espalha por muitos coeficientes, ou seja, textura mais irregular e menos concentrada em poucas bordas.

Ambas são a média das sub-bandas LH, HL e HH de todos os níveis. O CSV de features guarda também o valor de cada sub-banda isolada (`energy_L1_LH`, `entropy_L2_HH`, ...) e da banda de aproximação, para quem quiser investigar em qual escala a diferença aparece.

## Como ler o dashboard

Para cada par, o painel mostra a original, a sintética e a máscara usada no cálculo. Em seguida, para energia e entropia, ele responde visualmente a duas perguntas independentes:

1. **Ela preservou o par?** O percentual mostra quanto a feature mudou em relação à CT original que condicionou a geração. Também aparece a distância em IQR, a escala de variabilidade observada nas CTs reais.
2. **Ela continua plausível?** A faixa verde mostra a região esperada nas originais validadas; os pontos azul e laranja são, respectivamente, a original e a sintética do par.

O cartão de decisão traduz a combinação dessas perguntas em uma orientação simples: aprovada, aprovada com observação, revisão manual ou não aprovada.

## Arquivos de saída

- `case_validation.csv`: todos os candidatos, aceitos ou descartados, e a razão.
- `features_wavelet.csv`: uma linha por imagem, original ou sintética, com energia e entropia agregadas e por sub-banda.
- `reference_originals.csv`: média, mediana, desvio padrão, IQR, percentis e limites esperados das originais aceitas.
- `comparison_results.csv`: diferença por par, comparação com a referência e a classificação inicial.
- `figures/reference_distributions.png`: histogramas das originais; média e mediana indicam o centro e os limites Tukey mostram a faixa operacional esperada.
- `figures/original_vs_synthetic.png`: cada ponto é um par. Pontos próximos da diagonal preservam melhor a feature do caso original.
- `figures/example_case.png`: exemplo da original, sintética e máscara que entrou no cálculo.

## Como interpretar a classificação inicial

- `approved`: a sintética está próxima do par e dentro da referência.
- `approved_with_observation`: original e sintética são um caso raro de forma compatível.
- `manual_review`: há mudança relevante em relação ao par ou uma situação não canônica; fica fora do augmentation até revisão.
- `reject`: sintética distante do par e fora da referência; não entra no augmentation.

O limiar de proximidade do par começa em **1 IQR** da distribuição de originais para cada feature. Ele é configurável e deve ser revisado quando houver avaliação humana de exemplos aprovados, revisados e rejeitados.
