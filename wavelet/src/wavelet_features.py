"""Wavelet mask based: energia e entropia por sub-banda, calculadas apenas dentro do pulmão."""
from __future__ import annotations

import numpy as np
import pywt


PATCH_SIZE = 16
PATCH_STRIDE = 4
MIN_TISSUE_FRACTION = 0.9


def _patch_positions(mask: np.ndarray, size: int, stride: int, min_tissue: float) -> list[tuple[int, int]]:
    """Janelas deslizantes cuja fração de tecido pulmonar atinge o mínimo exigido.

    Exigir 100% de tecido descarta toda a periferia: uma janela rígida não cabe
    na anatomia fina dos cortes de ápice e base, e a cobertura cai para 13% nos
    piores casos. Aceitar uma fração deixa a borda entrar no cálculo.
    """
    integral = np.pad(mask.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    needed = size * size * min_tissue
    positions = []
    for row in range(0, mask.shape[0] - size + 1, stride):
        for col in range(0, mask.shape[1] - size + 1, stride):
            covered = (integral[row + size, col + size] - integral[row, col + size]
                       - integral[row + size, col] + integral[row, col])
            if covered >= needed:
                positions.append((row, col))
    return positions


def _subband_energy_entropy(band: np.ndarray) -> tuple[float, float]:
    """Energia média e entropia normalizada dos coeficientes de uma sub-banda.

    A entropia é dividida por log2(N) porque a entropia de Shannon crua cresce
    com o número de coeficientes: sem normalizar, ela mede o tamanho da ROI
    (correlação de 0,95 com a área pulmonar neste dataset) e não a textura.
    """
    values = band.ravel()
    energy = float(np.mean(values**2))
    magnitude = np.abs(values)
    total = magnitude.sum()
    if total <= 0 or values.size < 2:
        return energy, 0.0
    probability = magnitude[magnitude > 0] / total
    entropy = float(-(probability * np.log2(probability)).sum())
    return energy, entropy / np.log2(values.size)


def wavelet_energy_entropy(
    image: np.ndarray,
    mask: np.ndarray,
    clip_low: float,
    clip_high: float,
    wavelet: str = "haar",
    level: int = 2,
    patch_size: int = PATCH_SIZE,
    patch_stride: int = PATCH_STRIDE,
    min_tissue: float = MIN_TISSUE_FRACTION,
) -> dict[str, float]:
    """Decompõe o pulmão em janelas sobrepostas e resume energia/entropia dos detalhes.

    Mede cada janela isoladamente em vez de uma ROI única: a DWT exige grade
    retangular, e uma ROI única obriga a preencher tudo que não é pulmão — 48%
    da caixa delimitadora neste dataset, o que amortece a energia em ~47%.
    """
    if clip_high <= clip_low:
        raise ValueError("faixa de clipping inválida")
    if not mask.any():
        raise ValueError("a ROI não tem pixels marcados")
    max_level = pywt.dwtn_max_level((patch_size, patch_size), wavelet)
    if max_level < 1:
        raise ValueError(f"janela de {patch_size}px é pequena demais para a wavelet {wavelet}")
    used_level = min(level, max_level)

    clipped = np.clip(image, clip_low, clip_high)
    normalised = (clipped - clip_low) / (clip_high - clip_low)
    positions = _patch_positions(mask, patch_size, patch_stride, min_tissue)
    if not positions:
        raise ValueError("nenhuma janela satisfaz a fração mínima de tecido")

    energies: dict[str, list[float]] = {}
    entropies: dict[str, list[float]] = {}
    covered = np.zeros(mask.shape, dtype=bool)
    for row, col in positions:
        covered[row:row + patch_size, col:col + patch_size] = True
        coeffs = pywt.wavedec2(normalised[row:row + patch_size, col:col + patch_size], wavelet=wavelet, level=used_level)
        for depth, bands in enumerate(coeffs[1:], start=1):
            for name, band in zip(("LH", "HL", "HH"), bands):
                energy, entropy = _subband_energy_entropy(band)
                energies.setdefault(f"L{depth}_{name}", []).append(energy)
                entropies.setdefault(f"L{depth}_{name}", []).append(entropy)

    result: dict[str, float] = {}
    for key in energies:
        result[f"energy_{key}"] = float(np.mean(energies[key]))
        result[f"entropy_{key}"] = float(np.mean(entropies[key]))
    result["energy"] = float(np.mean([value for values in energies.values() for value in values]))
    result["entropy"] = float(np.mean([value for values in entropies.values() for value in values]))
    result["patch_count"] = float(len(positions))
    result["tissue_coverage"] = float((covered & mask).sum() / mask.sum())
    result["levels_used"] = float(used_level)
    return result
