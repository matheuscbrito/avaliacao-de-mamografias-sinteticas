"""GLCM com régua robusta compartilhada para CT pulmonar."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np


OFFSETS = ((0, 1), (-1, 1), (-1, 0), (-1, -1))


@dataclass(frozen=True)
class RobustScale:
    """Régua calibrada exclusivamente com pixels reais da ROI."""

    q1: float
    q3: float
    iqr: float
    clip_low: float
    clip_high: float
    tukey_k: float = 3.0
    levels: int = 32

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def calibrate_robust_scale(real_roi_pixels: np.ndarray, *, tukey_k: float = 3.0, levels: int = 32) -> RobustScale:
    """Calcula as cercas externas de Tukey nos pixels reais, nunca nos sintéticos."""
    values = np.asarray(real_roi_pixels, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("não há pixels reais válidos para calibrar a régua")
    if tukey_k < 0 or levels < 2:
        raise ValueError("tukey_k deve ser não negativo e levels deve ser >= 2")
    q1, q3 = (float(x) for x in np.percentile(values, (25, 75)))
    iqr = q3 - q1
    return RobustScale(q1, q3, iqr, q1 - tukey_k * iqr, q3 + tukey_k * iqr, tukey_k, levels)


def quantize(image: np.ndarray, scale: RobustScale) -> np.ndarray:
    """Aplica clipping e converte a imagem para a régua comum de 0 a levels - 1."""
    if scale.clip_high <= scale.clip_low:
        raise ValueError("régua inválida: teto deve ser maior que piso")
    clipped = np.clip(np.asarray(image, dtype=float), scale.clip_low, scale.clip_high)
    values = np.floor((clipped - scale.clip_low) / (scale.clip_high - scale.clip_low) * (scale.levels - 1))
    return np.clip(values, 0, scale.levels - 1).astype(np.int16)


def _aligned_pairs(image: np.ndarray, mask: np.ndarray, dy: int, dx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = image.shape
    y0, y1 = max(0, -dy), min(rows, rows - dy)
    x0, x1 = max(0, -dx), min(cols, cols - dx)
    first, second = image[y0:y1, x0:x1], image[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    valid = mask[y0:y1, x0:x1] & mask[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    return first, second, valid


def extract_glcm(image: np.ndarray, eroded_roi: np.ndarray, scale: RobustScale) -> dict[str, float]:
    """Extrai contraste e energia por quatro direções, só de vizinhos dentro da ROI.

    A matriz de cada direção é simétrica e normalizada antes de calcular as
    métricas; o resultado final é a média das quatro direções.
    """
    image = np.asarray(image)
    mask = np.asarray(eroded_roi, dtype=bool)
    if image.ndim != 2 or image.shape != mask.shape:
        raise ValueError("imagem e ROI devem ser matrizes 2D de mesmo tamanho")
    if not mask.any():
        raise ValueError("ROI erodida vazia")
    tones = quantize(image, scale)
    gray = np.arange(scale.levels, dtype=float)
    squared_difference = (gray[:, None] - gray[None, :]) ** 2
    contrasts, energies, pair_counts = [], [], []
    for dy, dx in OFFSETS:
        first, second, valid = _aligned_pairs(tones, mask, dy, dx)
        if not valid.any():
            continue
        matrix = np.zeros((scale.levels, scale.levels), dtype=float)
        np.add.at(matrix, (first[valid], second[valid]), 1)
        matrix += matrix.T
        matrix /= matrix.sum()
        contrasts.append(float((matrix * squared_difference).sum()))
        energies.append(float((matrix ** 2).sum()))
        pair_counts.append(int(valid.sum()))
    if not contrasts:
        raise ValueError("ROI não contém pares vizinhos válidos")
    roi_tones = tones[mask]
    return {
        "glcm_contrast": float(np.mean(contrasts)),
        "glcm_energy": float(np.mean(energies)),
        "saturation_tone_0_pct": float(np.mean(roi_tones == 0) * 100),
        "saturation_tone_31_pct": float(np.mean(roi_tones == scale.levels - 1) * 100),
        "directions_used": float(len(contrasts)),
        "mean_valid_neighbor_pairs": float(np.mean(pair_counts)),
    }
