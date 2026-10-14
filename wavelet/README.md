# Wavelet: validação de textura em CT pulmonar sintética

Este projeto avalia se uma CT pulmonar sintética preserva a textura da original
que a condicionou e continua plausível perante as CTs reais validadas do mesmo
dataset. É o irmão do MVP de GLCM (`../glcm`): mesmo fluxo, outra família de
features.

A wavelet decompõe a imagem em sub-bandas de frequência e escala. Ela enxerga a
textura em várias escalas ao mesmo tempo e diz **o que** foi perdido (borrou,
perdeu contraste, estourou, ruído). O **brilho** é medido por um eixo separado,
a densidade, porque a wavelet de detalhe não enxerga brilho.

> **Documentação completa:** [`../documentacao/WAVELET.md`](../documentacao/WAVELET.md)
> explica o pipeline, todas as regras (validação, referência, decisão, densidade,
> veredito de textura) e o mapa por região. Este README só diz como rodar e onde
> está cada coisa.

## Fluxo

```text
original + sintética + máscara
              ↓
validação do caso e erosão leve da máscara (ROI pulmonar)
              ↓
faixa de referência P1–P99 (pixels pulmonares das originais aceitas)
              ↓
densidade: o pulmão ficou claro ou escuro demais?
wavelet (Haar, 2 níveis, imagem inteira): energia e entropia nos coeficientes do pulmão
              ↓
referência das originais → comparação com o par e com a referência
              ↓
diagnóstico: LL, energia fina, energia grossa, razão → tipo de defeito
mapa por região: o mesmo diagnóstico bloco a bloco + correlação espacial
              ↓
CSVs + gráficos + dashboard
```

## Instalação

A partir desta pasta, em um ambiente virtual:

```bash
pip install -r requirements.txt
```

O dataset atual é plano: cada original `caso.tiff` precisa ter
`caso_generate.tiff` e `caso_mask.tiff`. Veja `data/README.md` para o formato
alternativo (`original/`, `synthetic/`, `masks/`).

**Dataset padrão:** `../dados/Dsr`. O Dsr não traz máscaras; elas vêm de
`../dados/Dgen` (as originais são idênticas). O Dgen entra só como fonte de
exemplos quebrados.

## Como rodar cada análise

Todos os comandos partem de `wavelet/`. Os resultados e a leitura de cada um
ficam em `../resultados/<pasta>/resumo.md`.

| Análise | Comando | Resultado |
|---|---|---|
| Pipeline básico (CSVs, referência, classificação do par) | `python -m src.run_pipeline_fullimage --data-dir ../dados/Dgen --output-dir outputs` | `outputs/` (é onde o dashboard lê) |
| Resposta da wavelet ao brilho, contraste e lavagem | `python -m src.wavelet_brightness --dgen-dir ../dados/Dgen --dsr-dir ../dados/Dsr --output-dir ../resultados/wavelet_brilho` | `resultados/wavelet_brilho/` |
| Assinatura de cada defeito (LL, fina, grossa, razão) | `python -m src.wavelet_diagnostico --data-dir ../dados/Dsr --mask-dir ../dados/Dgen --examples-dir ../dados/Dgen --output-dir ../resultados/wavelet_diagnostico` | `resultados/wavelet_diagnostico/` |
| Densidade + veredito da wavelet (seis jeitos de clarear) | `python -m src.densidade_wavelet --data-dir ../dados/Dsr --mask-dir ../dados/Dgen --examples-dir ../dados/Dgen --output-dir ../resultados/densidade_wavelet` | `resultados/densidade_wavelet/` |
| Mapa por região (blocos de 16 px) | `python -m src.mapa_regioes --data-dir ../dados/Dsr --mask-dir ../dados/Dgen --examples-dir ../dados/Dgen --block 16 --output-dir ../resultados/mapa_regioes` | `resultados/mapa_regioes/` |
| Teste de perturbação dos quatro eixos | `python -m src.perturbation_test --data-dir ../dados/Dgen --output-dir ../resultados/teste_perturbacao_wavelet` | `resultados/teste_perturbacao_wavelet/` |
| Comparação Dgen × Dsr (histórica) | `python -m src.compare_generations --dgen-dir ../dados/Dgen --dsr-dir ../dados/Dsr --output-dir ../resultados/comparacao_Dgen_Dsr_wavelet` | `resultados/comparacao_Dgen_Dsr_wavelet/` |
| Figuras do report (exemplo de brilho sem resposta da wavelet; curva de brilho) | `python -m src.figuras_report --data-dir ../dados/Dsr --mask-dir ../dados/Dgen --examples-dir ../dados/Dgen --case-id "1.000000-P4P105S300I00009 Gated 50.0A-40237-1-081" --sweep-csv ../resultados/wavelet_brilho/varredura_por_caso.csv --output-dir ../slides/imagens` | `slides/imagens/` |
| Dashboard | `streamlit run dashboard/app.py` | lê os arquivos prontos |

O ambiente do repositório fica em `../.venv`; nos resumos os comandos aparecem
como `.venv/bin/python -m src.<script>`.

O pipeline básico ainda roda no Dgen: ele espera original, sintética e máscara
na mesma pasta, e o Dsr não traz máscaras. As outras análises usam o Dsr com
`--mask-dir ../dados/Dgen`. O `figuras_report.py` depende do CSV gerado pelo
`wavelet_brightness.py`, então rode esse antes.

### Parâmetros comuns

- `--wavelet` (padrão `haar`): família wavelet, qualquer nome do PyWavelets
  (`db2`, `sym4`, `coif1`…). Trocar a família muda os valores de energia;
  mantenha a mesma ao comparar resultados.
- `--level` (padrão `2`): níveis de decomposição. Se a imagem for pequena
  demais, o script usa o máximo possível e registra em `levels_used`.
- `--min-tissue` (padrão `0.9`): fração mínima de pulmão para um coeficiente ou
  bloco entrar na conta.
- `--erosion-radius` (padrão `2`): erosão da máscara, em pixels.
- `--pair-iqr-threshold` (padrão `1.0`): distância máxima do par, em IQRs da
  referência.
- `--block` (padrão `16`, só no mapa): lado do bloco em px, múltiplo de 4.

## O que cada script faz

**Núcleo**

- `src/load_data.py`: encontra os três arquivos de cada caso e abre as imagens.
- `src/validate_cases.py`: aceita só pares que correspondem e máscaras com dois pulmões utilizáveis; grava os motivos de descarte.
- `src/wavelet_features_fullimage.py`: **método adotado.** Decompõe a imagem inteira uma vez e mede energia e entropia nos coeficientes cobertos pela máscara.
- `src/build_reference.py`: distribuição de referência das originais aceitas (média, IQR, Tukey).
- `src/compare_images.py`: compara a sintética com o par e com a referência e dá a classificação.
- `src/run_pipeline_fullimage.py`: orquestra o fluxo do método adotado e salva CSVs e figuras.
- `src/density_features.py`: eixo de densidade (pulmão claro ou escuro demais).

**Diagnóstico**

- `src/wavelet_brightness.py`: como energia, entropia e LL respondem a brilho, contraste e lavagem.
- `src/wavelet_diagnostico.py`: as leituras LL, fina, grossa e razão; assinatura de oito defeitos simulados.
- `src/densidade_wavelet.py`: seis jeitos de clarear o pulmão e a regra do veredito de textura (`wavelet_verdict`).
- `src/mapa_regioes.py`: o mesmo diagnóstico por bloco, com correlação espacial (Spearman).
- `src/perturbation_test.py`: defeitos conhecidos nas originais, medidos nos quatro eixos.
- `src/nyquist_excess.py`: estimador NEx (de [gabrielbringel/nyquist-excess](https://github.com/gabrielbringel/nyquist-excess)), usado só no teste de perturbação.

**Comparações e figuras**

- `src/compare_generations.py`: compara Dgen e Dsr nos mesmos cortes (histórico).
- `src/figuras_report.py`: figuras para o relatório e os slides (exemplo de brilho sem resposta da wavelet; curva de brilho).
- `src/wavelet_features.py`, `src/run_pipeline.py`: método antigo (janelas 16×16), mantido sem alteração para comparação. O `run_pipeline.py` também fornece `write_csv`, usado pelos outros scripts.
- `src/compare_patch_vs_fullimage.py`: experimento que compara os dois métodos (relatório em `outputs/patch_vs_fullimage/REPORT.md`, pasta ignorada pelo git).
- `src/visualize_wavelet_example.py`: mapa de energia espacial dos dois métodos, só para inspeção visual.

**Dashboard**

- `dashboard/app.py`: lê os arquivos prontos; não recalcula features.
- `dashboard/pages/1_Dgen_x_Dsr.py`: página da comparação Dgen × Dsr, a partir de `../resultados/comparacao_Dgen_Dsr_wavelet`.

> O dashboard ainda não mostra as colunas de densidade, o diagnóstico nem o
> mapa por região. Essas saídas estão nas pastas de `../resultados/`.

## Como a ROI entra na wavelet

A máscara é erodida (raio 2 px) para tirar a borda do pulmão. A imagem inteira
é decomposta uma vez, e a máscara é reduzida ao tamanho de cada sub-banda:
cada coeficiente cobre um bloco de 2×2 px (nível fino) ou 4×4 px (nível
grosso). Só entram os coeficientes cujo bloco tem ao menos 90% de pulmão. Assim
o fundo e a borda nítida entre pulmão e fora não viram o padrão dominante de
alta frequência.

## As features

- **Energia** (`energy`): média dos coeficientes de detalhe ao quadrado. Cresce com estrutura fina, bordas, reticulação e contraste.
- **Entropia** (`entropy`): dispersão dos coeficientes de detalhe, normalizada por `log2(N)`. Cresce quando a energia se espalha por muitos coeficientes.
- **LL**: média da banda de aproximação, na escala da faixa P1–P99 (0 = P1, 1 = P99). Única leitura da wavelet que acompanha o brilho.
- **Fina ÷ grossa**: razão entre a energia de 2 px e a de 4 px. Abaixo de 1, o detalhe fino se perdeu mais (borramento).

`energy` e `entropy` são a média das sub-bandas LH, HL e HH de todos os níveis.
O CSV guarda também cada sub-banda isolada (`energy_L1_LH`, `entropy_L2_HH`…).

> **Cuidado com os nomes:** no código e nos CSVs, `L1` é o nível **mais grosso**
> (4 px) e `L2` o **mais fino** (2 px). Vem da ordem de `pywt.wavedec2`.

## Como ler o dashboard

Para cada par, o painel mostra a original, a sintética e a máscara usada. Para
energia e entropia, ele responde a duas perguntas independentes:

1. **Ela preservou o par?** O percentual mostra quanto a feature mudou em relação à original. Aparece também a distância em IQR, a escala de variabilidade das CTs reais.
2. **Ela continua plausível?** A faixa verde mostra a região esperada nas originais validadas; os pontos azul e laranja são a original e a sintética do par.

O cartão de decisão traduz as duas respostas: aprovada, aprovada com
observação, revisão manual ou não aprovada.

## Arquivos de saída do pipeline básico

- `case_validation.csv`: todos os candidatos, aceitos ou descartados, e a razão.
- `features_wavelet.csv`: uma linha por imagem, com energia e entropia agregadas e por sub-banda.
- `reference_originals.csv` e `.json`: estatísticas das originais aceitas, mais a faixa P1–P99 e os parâmetros usados.
- `comparison_results.csv`: diferença por par, comparação com a referência e a classificação inicial.
- `figures/reference_distributions.png`: histogramas das originais, com média, mediana e limites de Tukey.
- `figures/original_vs_synthetic.png`: cada ponto é um par; perto da diagonal preserva melhor a feature.
- `figures/example_case.png`: original, sintética e máscara usada.

Os outros scripts gravam em `../resultados/<pasta>/` um `resumo.md` (leitura),
`resumo.json`, CSVs por caso e `figuras/`.

## Como interpretar a classificação

- `approved`: a sintética está próxima do par e dentro da referência.
- `approved_with_observation`: original e sintética são um caso raro de forma compatível.
- `manual_review`: há mudança relevante em relação ao par ou uma situação não canônica; fica fora do augmentation até revisão.
- `reject`: sintética distante do par e fora da referência; não entra no augmentation.

O limiar de proximidade do par começa em **1 IQR** da distribuição de originais
para cada feature. Ele é configurável e deve ser revisado quando houver
avaliação humana de exemplos aprovados, revisados e rejeitados.

A guarda de densidade (`*_status_with_density`) manda para `manual_review` os
cortes aprovados pela textura cuja densidade não está `ok`; um `reject`
continua `reject`. A regra completa e o veredito de textura por
região estão detalhados em [`../documentacao/WAVELET.md`](../documentacao/WAVELET.md).
Os limites do veredito (80–125% de energia, razão 0,9–1,1) são exploratórios,
não validados.
