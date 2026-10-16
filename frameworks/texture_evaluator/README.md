# Avaliador de textura

O MVP reúne GLCM e Wavelet nos mesmos pares e na mesma ROI erodida. Ele salva tabela única e prévias visuais do par real/sintético/ROI.

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

O brilho tem duas leituras lado a lado: o filtro do contrato (`brightness_status`, por Δμ e Sat31) e o eixo de densidade da Wavelet (`density_status`, `density_median_shift_iqr`). Como as duas convivem ainda será decidido. A terceira métrica está reservada, sem implementação especulativa.

## Decisão final

Cada corte sai com `final_status` (`aprovada`, `revisar` ou `rejeitar`) e os motivos em `final_reasons`. As regras ficam em [`decision.py`](decision.py):

1. **Brilho:** só o filtro do contrato rejeita; a densidade manda para revisão.
2. **Quantidade de detalhe:** contraste GLCM e energia fina Wavelet devem concordar (80–125% da original); se discordarem, o corte vai para revisão.
3. **Tipo de defeito:** veredito Wavelet diferente de "textura preservada" manda para revisão.

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
- Calibrar os limites de textura e da checagem do par com avaliação visual de alguém da área.
- Integrar a terceira métrica (NEx) e avaliar a qualidade dos nódulos. A origem da assinatura NEx no Dsr já foi localizada na difusão/amostragem, não no AEKL ([`resultados/origem_assinatura_nex/resumo.md`](../../resultados/origem_assinatura_nex/resumo.md)); falta saber se ela prejudica o data augmentation. A faixa real do NEx na ROI do MVP está em [`resultados/nex_referencia_roi/resumo.md`](../../resultados/nex_referencia_roi/resumo.md) (script `nex_referencia.py`).
