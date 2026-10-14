"""Estimador experimental `nyquist_excess` (NEx), usado só no teste de perturbação.

Proveniência: cópia de `src/nyquist_excess/metric.py` do repositório
https://github.com/gabrielbringel/nyquist-excess (Gabriel Bringel, commit
f7177c1dd40507fd1653c8dc00132ceb0b952670). A aritmética é a mesma; só a
validação de entrada e as docstrings foram reduzidas. Para a explicação
completa, limitações e o teste cego em TC, ver aquele repositório.

Resumo: perfil médio de |diferença entre vizinhos| em cada eixo, espectro de
potência com janela de Hann, e log10(pico nos 3 bins mais perto de Nyquist /
mediana do resto). O escore é o máximo dos dois eixos. Por construção é
invariante a `a·I + b`, então não deve reagir a brilho nem a ganho.
"""
from __future__ import annotations

import numpy as np

N_TAIL_BINS = 3


def _gradient_profile_spectrum(tile: np.ndarray, axis: int) -> np.ndarray:
    g = np.abs(np.diff(np.asarray(tile, dtype=np.float64), axis=axis)).mean(axis=1 - axis)
    return np.abs(np.fft.rfft((g - g.mean()) * np.hanning(g.size))) ** 2


def _peak_over_background(spec: np.ndarray, index: int, masked: np.ndarray) -> float:
    bg = float(np.median(spec[1:][~masked[1:]]))
    i = min(int(index), spec.size - 1)
    peak = float(spec[max(0, i - 2):min(i + 3, spec.size)].max())
    return float(np.log10(max(peak, 1e-300) / max(bg, 1e-300)))


def nyquist_excess(tile: np.ndarray) -> float:
    """NEx em décadas: máximo entre os eixos x e y."""
    if tile.ndim != 2 or min(tile.shape) < 8:
        raise ValueError(f"esperado tile 2D com pelo menos 8 px por eixo, veio {tile.shape}")
    scores = []
    for axis in (1, 0):
        spec = _gradient_profile_spectrum(tile, axis)
        masked = np.zeros(spec.size, dtype=bool)
        masked[-N_TAIL_BINS:] = True
        scores.append(_peak_over_background(spec, spec.size - 1, masked))
    return max(scores)
