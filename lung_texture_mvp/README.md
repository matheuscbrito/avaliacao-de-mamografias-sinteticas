# MVP de validação de textura pulmonar com GLCM

Este projeto avalia se uma CT pulmonar sintética preserva a textura da original que a condicionou e continua plausível perante as CTs reais validadas do mesmo dataset.

## Fluxo

```text
original + sintética + máscara
              ↓
validação do caso e erosão leve da máscara
              ↓
GLCM dentro do pulmão: contraste e energia
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

No Mac do Lucas, o ambiente virtual já está no repositório principal. Se os
comandos globais `python`, `pip` ou `streamlit` não forem encontrados, use:

```bash
"/Users/ltodaro/dev/metricas-avaliacao/.venv/bin/python" -m pip install -r requirements.txt
"/Users/ltodaro/dev/metricas-avaliacao/.venv/bin/python" -m src.run_pipeline --data-dir "/Users/ltodaro/dev/metricas-avaliacao/upscale_test_new_lung_Dgen"
"/Users/ltodaro/dev/metricas-avaliacao/.venv/bin/python" -m streamlit run dashboard/app.py
```

O dataset atual é plano: cada original `caso.tiff` precisa ter `caso_generate.tiff` e `caso_mask.tiff`. O MVP também aceita a estrutura `data/original`, `data/synthetic` e `data/masks`.

## O que cada script faz

- `src/load_data.py`: encontra os três arquivos de cada caso e abre as imagens.
- `src/validate_cases.py`: aceita somente pares que correspondem e máscaras com dois pulmões utilizáveis; grava os motivos de descarte.
- `src/glcm_features.py`: calcula contraste e energia usando apenas pares de pixels dentro da máscara erodida.
- `src/build_reference.py`: cria a distribuição de referência das originais aceitas.
- `src/compare_images.py`: compara a sintética tanto com a original do par quanto com a referência.
- `src/run_pipeline.py`: orquestra tudo e salva CSVs e figuras.
- `dashboard/app.py`: lê os arquivos prontos; não recalcula features.

## Como ler o dashboard

Para cada par, o painel mostra a original, a sintética e a máscara usada no
cálculo. Em seguida, para contraste e energia, ele responde visualmente a duas
perguntas independentes:

1. **Ela preservou o par?** O percentual mostra quanto a feature mudou em
   relação à CT original que condicionou a geração. Também aparece a distância
   em IQR, a escala de variabilidade observada nas CTs reais.
2. **Ela continua plausível?** A faixa verde mostra a região esperada nas
   originais validadas; os pontos azul e laranja são, respectivamente, a
   original e a sintética do par.

O cartão de decisão traduz a combinação dessas perguntas em uma orientação
simples: aprovada, aprovada com observação, revisão manual ou não aprovada.

## Arquivos de saída

- `case_validation.csv`: todos os candidatos, aceitos ou descartados, e a razão.
- `features_glcm.csv`: uma linha por imagem, original ou sintética, com contraste e energia.
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
