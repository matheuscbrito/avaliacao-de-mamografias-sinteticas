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

GLCM mede contraste e energia. Wavelet mede energia e entropia de detalhe; LL é registrada para brilho. A terceira métrica está reservada, sem implementação especulativa.

Depois da execução, abra `outputs/mvp/report/index.html` no navegador. Ele explica cada número, mostra os gráficos pareados real × sintética e traz uma ficha visual para cada corte analisado.
