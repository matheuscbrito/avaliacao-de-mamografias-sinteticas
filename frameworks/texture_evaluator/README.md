# Avaliador de textura

O MVP reúne GLCM e Wavelet nos mesmos pares e na mesma ROI erodida. Ele salva tabela única e prévias visuais do par real/sintético/ROI.

```bash
python -m frameworks.texture_evaluator.run_mvp --data-dir dados/Dsr --output-dir outputs/mvp
```

GLCM mede contraste e energia. Wavelet mede energia e entropia de detalhe; LL é registrada para brilho. A terceira métrica está reservada, sem implementação especulativa.
