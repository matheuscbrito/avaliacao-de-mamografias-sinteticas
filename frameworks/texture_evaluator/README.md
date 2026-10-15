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

Depois da execução, abra `outputs/mvp/report/index.html` no navegador. Ele explica cada número, mostra os gráficos pareados real × sintética e traz uma ficha visual para cada corte analisado.
