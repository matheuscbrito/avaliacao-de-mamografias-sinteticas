# Avaliador de textura

O MVP reúne GLCM, Wavelet e NEx nos mesmos pares e na mesma ROI erodida. Ele salva tabela única e prévias visuais do par real/sintético/ROI.

Na raiz do repositório, execute:

```bash
.venv/bin/python -m frameworks.texture_evaluator.run_mvp \
  --data-dir dados/Dsr --mask-dir dados/Dgen --output-dir outputs/mvp
```

Se você estiver dentro de `frameworks/texture_evaluator/`, volte primeiro para
a raiz com `cd ../..`. O ambiente local `.venv` contém as dependências da
Wavelet; não há garantia de um comando global chamado `python`.

GLCM mede contraste e energia. Wavelet mede energia e entropia de detalhe; LL é registrada para brilho. A Wavelet também dá um veredito do tipo de mudança de textura (`wavelet_verdict`: borrou, ruído, perdeu contraste…), a partir da energia fina e grossa em % da original (`wavelet_fine_pct`, `wavelet_coarse_pct`, `wavelet_ratio`; limites exploratórios, ver [`docs/wavelet.md`](../../docs/wavelet.md)).

GLCM e Wavelet usam a **mesma régua** de clipping: Tukey k=3 calibrada nos pixels reais (protocolo GLCM). Com ela, as duas ficam cegas a brilho puro; a régua P1–P99, usada antes pela Wavelet, fazia a energia GLCM e a entropia Wavelet reagirem ao escurecimento. Teste e números em [`resultados/regua_unica/resumo.md`](../../resultados/regua_unica/resumo.md); para refazer:

```bash
.venv/bin/python -m frameworks.texture_evaluator.teste_regua \
  --data-dir dados/Dsr --mask-dir dados/Dgen --output-dir resultados/regua_unica
```

O brilho tem duas leituras lado a lado: o filtro do contrato (`brightness_status`, por Δμ e Sat31) e o eixo de densidade da Wavelet (`density_status`, `density_median_shift_iqr`). Como as duas convivem ainda será decidido. A banda LL da Wavelet também dá uma leitura de brilho (ver abaixo). O NEx mede excesso ou falta de estrutura perto do limite de resolução, na mesma ROI.

## Decisão final

Cada corte sai com `final_status` (`aprovada`, `revisar` ou `rejeitar`) e os motivos em `final_reasons`. As regras ficam em [`decision.py`](decision.py):

1. **Brilho:** só o filtro do contrato rejeita; a densidade manda para revisão.
2. **Quantidade de detalhe:** contraste GLCM e energia fina Wavelet devem concordar (80–125% da original); se discordarem, o corte vai para revisão.
3. **Tipo de defeito:** veredito Wavelet diferente de "textura preservada" manda para revisão.
4. **NEx:** fora de P5–P95 das originais manda para revisão; poucas janelas na ROI é inconclusivo.

## Classificação por técnica (painel)

O painel lê cada técnica separadamente, sem mudar o veredito final acima. Todas as métricas com régua de IQR usam a mesma regra do filtro de brilho: a diferença sintética − original, dividida pelo IQR daquela métrica entre as CTs reais de referência (calculado na primeira versão e congelado no histórico).

**Veredito de textura (aba de textura do painel).** GLCM, Wavelet e NEx são classificadas cada uma por conta própria, e o veredito de textura é a **pior das três** (`decision.texture_status`). O brilho não entra.

| Técnica | Métricas | Regra |
|---|---|---|
| GLCM | contraste, energia | diferença sintética − original em IQR da métrica entre as CTs reais (`glcm_iqr`): até 1 **aprovada**, 1–2 **revisão**, acima de 2 **reprovada**; a pior métrica decide |
| Wavelet | 2 px, 4 px, entropia | a mesma regra (`wavelet_iqr`); o "tipo" (borrou, ruído…) descreve, não decide |
| NEx | NEx na ROI | fora de P5–P95 das CTs reais = **revisão**; poucas janelas = inconclusivo, não pesa |

**Brilho segundo a Wavelet (modo de brilho do painel).** A banda LL (média local de 4×4 px, só o brilho) recebe a mesma regra: Δ = LL da sintética − LL da original, em IQR da LL entre as CTs reais (`ll_iqr`; colunas `wavelet_ll_z` e `wavelet_ll_status`). É só leitura, ao lado do filtro oficial (Δμ, Sat31), e não muda nenhuma decisão.

Colunas: `glcm_*_z`, `glcm_*_status`, `glcm_status`, `wavelet_*_z`, `wavelet_*_status`, `wavelet_status` e `nex_status`. O painel mostra uma matriz "quem disse o quê" (corte × técnica) e uma aba por técnica com os gráficos de cada métrica. A reprovação vale só na leitura de textura; o veredito final acima continua mandando textura para revisão. No Dsr: 8 preservadas, 29 revisão, 0 reprovadas; no Dgen: 2, 26 e 9. A regra é branda porque a textura varia muito entre cortes reais (o IQR do 2 px é ~47% da mediana).

A validação com defeitos conhecidos está em [`resultados/teste_decisao/resumo.md`](../../resultados/teste_decisao/resumo.md):

```bash
.venv/bin/python -m frameworks.texture_evaluator.teste_decisao \
  --data-dir dados/Dsr --mask-dir dados/Dgen --output-dir resultados/teste_decisao
```

## Placar por versão do gerador

Para acompanhar o modelo ao longo das iterações. Cada versão é a pasta de saída de uma iteração, gerada das **mesmas originais**. O placar guarda um histórico: cada versão é avaliada uma vez e as novas entram depois, sem recalcular as antigas. A primeira versão vira a referência das réguas.

**Interface** (adicionar versões, scores, mudanças, vereditos e tabela por corte):

```bash
.venv/bin/streamlit run frameworks/texture_evaluator/ui_placar.py --theme.base light --theme.font monospace
```

**Linha de comando** (mesmo histórico):

```bash
.venv/bin/python -m frameworks.texture_evaluator.placar --output-dir outputs/placar \
  --version v1=CAMINHO/DA/VERSAO1 --mask-dir dados/Dgen
.venv/bin/python -m frameworks.texture_evaluator.placar --output-dir outputs/placar \
  --version v2=CAMINHO/DA/VERSAO2
```

Cada par é montado pelo nome (`X.tiff` ↔ `X_generate.tiff` ↔ `X_mask.tiff`) e conferido em [`regional.py`](regional.py). No modo **mesma anatomia** (como o Dsr), um par com correlação < 0,5 com a original é rejeitado como suspeito (arquivo trocado?). No modo **só máscara** (como o Dgen: outro pulmão no mesmo contorno), o par só é conferido pelo nome, e o que compara ponto a ponto (área alterada, correlação espacial, mapa comparado com a original) fica como n/a. O placar e a interface separam **veredito final**, **brilho** e **textura**, com uma linha por técnica (ex.: "Wavelet 2 px (textura fina)"). Na interface, cada técnica mostra a imagem do que ela enxerga no corte escolhido, na original e em cada versão, e o mapa por região (blocos de 16 px) tem dois modos: valor de cada imagem (qualquer versão) e comparado com a original (só mesma anatomia).

`--remove NOME` tira uma versão. O resultado fica em `placar.md` (mediana de cada indicador com intervalo de 95%, vereditos e se cada versão melhorou, piorou ou não mudou em relação à anterior), `placar.csv`, `comparacao.csv` e `por_caso.csv`. Exemplo e validação em [`resultados/placar_versoes/resumo.md`](../../resultados/placar_versoes/resumo.md).

Depois da execução, abra `outputs/mvp/report/index.html` no navegador. Ele explica cada número, mostra os gráficos pareados real × sintética e traz uma ficha visual para cada corte analisado.

## Próximos passos

- **Comparar com pulmões reais, não só com a original do par.** Hoje brilho e textura são medidos contra a original de cada par. Para um gerador que cria um pulmão novo (modo só máscara, como o Dgen), essa original é outro pulmão no mesmo contorno, e o que compara ponto a ponto fica como n/a. Um modo "comparado com pulmões reais" mediria se cada corte (e cada bloco do mapa) cabe na faixa de variação das originais reais. Ele valeria igualmente para qualquer gerador, pareado ou não, e acabaria com o n/a. O modo atual, comparado com a original, ficaria como leitura extra quando a anatomia é a mesma.
- Calibrar os limites de textura e da checagem do par com avaliação visual de alguém da área. A regra de 1/2 IQR é branda; 1,5/2,5 IQR é a alternativa mais rígida a testar.
- Avaliar a qualidade dos nódulos e se a assinatura NEx prejudica o data augmentation. A origem da assinatura NEx no Dsr já foi localizada na difusão/amostragem, não no AEKL ([`resultados/origem_assinatura_nex/resumo.md`](../../resultados/origem_assinatura_nex/resumo.md)). A faixa real do NEx na ROI do MVP está em [`resultados/nex_referencia_roi/resumo.md`](../../resultados/nex_referencia_roi/resumo.md) (script `nex_referencia.py`).
