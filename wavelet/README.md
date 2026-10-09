# MVP de validação de textura pulmonar com wavelet

Este projeto avalia se uma CT pulmonar sintética preserva a textura da original que a condicionou e continua plausível perante as CTs reais validadas do mesmo dataset. É o irmão do MVP de GLCM (`../glcm`): mesmo fluxo, mesma leitura de resultados, outra família de features.

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
python -m src.run_pipeline_fullimage --data-dir ../dados/Dgen
streamlit run dashboard/app.py
```

O dataset atual é plano: cada original `caso.tiff` precisa ter `caso_generate.tiff` e `caso_mask.tiff`. O MVP também aceita a estrutura `data/original`, `data/synthetic` e `data/masks`.

Parâmetros específicos da wavelet:

- `--wavelet` (padrão `db4`): família wavelet, qualquer nome aceito pelo PyWavelets (`haar`, `db2`, `sym4`, `coif1`...).
- `--level` (padrão `2`): níveis de decomposição. Se a ROI for pequena demais para o nível pedido, o script usa o máximo possível e registra em `levels_used`.

## Comparação Dgen × Dsr

`src/compare_generations.py` roda o método de imagem inteira sobre as duas gerações (`../dados/Dgen` e `../dados/Dsr`) nos mesmos cortes da comparação GLCM. As máscaras do Dgen são reutilizadas depois de confirmar que as originais são idênticas nos dois conjuntos.

```bash
python -m src.compare_generations --dgen-dir ../dados/Dgen --dsr-dir ../dados/Dsr --output-dir ../resultados/comparacao_Dgen_Dsr_wavelet
```

O resultado e a leitura ficam em `../resultados/comparacao_Dgen_Dsr_wavelet/resumo.md`.

O dashboard tem uma página **Dgen x Dsr** que lê esses resultados. Nela dá para ver cada corte com a original e as duas sintéticas lado a lado, a decisão do framework para cada geração, e a energia por sub-banda para descobrir em qual escala o detalhe foi perdido. Os cortes podem ser ordenados pela vantagem de uma geração sobre a outra.

## Eixo de densidade

As features de textura (wavelet, GLCM e NEx) foram feitas para ignorar o brilho
médio. Por isso, um pulmão que ficou claro demais pode passar como textura boa.
`src/density_features.py` cobre esse ponto cego olhando só para o histograma
dos pixels dentro da máscara:

1. junta os pixels pulmonares de todas as originais e pega P1 e P99, a faixa
   normal de um pulmão real;
2. em cada imagem, mede a fração do pulmão acima do P99 (`frac_bright`), abaixo
   do P1 (`frac_dark`) e a posição da mediana dentro da faixa (`lung_median`,
   0 = P1, 1 = P99);
3. compara a sintética com **a sua própria original**: há cortes reais
   legitimamente escuros, e a sintética que os acompanha está certa.

O status (`*_density_status`) segue do mais grave ao mais leve:

- `too_bright` / `too_dark`: a sintética põe mais pulmão fora da faixa real do
  que a original, além do limite de Tukey das originais (saturação ou tecido
  trocado);
- `brighter` / `darker`: nada saiu da faixa, mas a mediana andou mais de 1 IQR
  das medianas das originais (o pulmão inteiro foi deslocado);
- `ok`.

A guarda não muda o status de textura. Quando a densidade não está `ok`, a
coluna `*_status_with_density` manda o corte para `manual_review`.

## Teste de perturbação

`src/perturbation_test.py` aplica defeitos conhecidos às **originais** (brilho,
contraste, gama, lavagem do pulmão, saturação, suavização e ruído). Depois mede
os quatro eixos: densidade, wavelet, GLCM (importada de `../glcm`) e NEx
(`src/nyquist_excess.py`, copiado de
[gabrielbringel/nyquist-excess](https://github.com/gabrielbringel/nyquist-excess)).
A mudança de cada métrica é dada em IQR entre as originais, então todos os eixos
ficam na mesma régua. O teste serve para conferir se cada eixo mede o que promete.

```bash
python -m src.perturbation_test --data-dir ../dados/Dgen --output-dir ../resultados/teste_perturbacao_wavelet
```

## O que cada script faz

- `src/load_data.py`: encontra os três arquivos de cada caso e abre as imagens.
- `src/validate_cases.py`: aceita somente pares que correspondem e máscaras com dois pulmões utilizáveis; grava os motivos de descarte.
- `src/wavelet_features.py`: recorta a ROI pulmonar, decompõe com DWT 2D e mede energia e entropia por sub-banda.
- `src/build_reference.py`: cria a distribuição de referência das originais aceitas.
- `src/compare_images.py`: compara a sintética tanto com a original do par quanto com a referência.
- `src/run_pipeline.py`: orquestra tudo e salva CSVs e figuras.
- `src/compare_generations.py`: compara Dgen e Dsr sobre os mesmos cortes e originais.
- `src/density_features.py`: eixo de densidade (pulmão claro ou escuro demais).
- `src/perturbation_test.py`: teste de perturbação controlada dos quatro eixos.
- `src/nyquist_excess.py`: estimador NEx, usado só no teste de perturbação.
- `dashboard/app.py`: lê os arquivos prontos; não recalcula features.
- `dashboard/pages/1_Dgen_x_Dsr.py`: página da comparação Dgen × Dsr, a partir de `../resultados/comparacao_Dgen_Dsr_wavelet`.

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
