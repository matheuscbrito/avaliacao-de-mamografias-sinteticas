"""GLCM mask based: contraste e energia, calculados apenas dentro do pulmão."""
from __future__ import annotations

import numpy as np


OFFSETS = ((0, 1), (-1, 1), (-1, 0), (-1, -1))
ANGLES = (0, 45, 90, 135)


def _aligned_pairs(image: np.ndarray, mask: np.ndarray, row: int, col: int):
    rows, cols = image.shape
    a_row_start, a_row_end = max(0, -row), min(rows, rows - row)
    a_col_start, a_col_end = max(0, -col), min(cols, cols - col)
    b_row_start, b_row_end = max(0, row), min(rows, rows + row)
    b_col_start, b_col_end = max(0, col), min(cols, cols + col)
    return (
        image[a_row_start:a_row_end, a_col_start:a_col_end],
        image[b_row_start:b_row_end, b_col_start:b_col_end],
        mask[a_row_start:a_row_end, a_col_start:a_col_end],
        mask[b_row_start:b_row_end, b_col_start:b_col_end],
    )


def glcm_contrast_energy(image: np.ndarray, mask: np.ndarray, clip_low: float, clip_high: float, levels: int = 32) -> dict[str, float]:
    """Mede GLCM simétrica para pares de vizinhos ambos dentro da máscara."""
    if clip_high <= clip_low:
        raise ValueError("faixa de clipping inválida")
    clipped = np.clip(image, clip_low, clip_high)
    quantized = np.floor((clipped - clip_low) / (clip_high - clip_low) * (levels - 1)).astype(np.uint8)
    gray = np.arange(levels, dtype=float)
    squared_difference = (gray[:, None] - gray[None, :]) ** 2
    contrasts, energies = [], []
    for row, col in OFFSETS:
        first, second, first_mask, second_mask = _aligned_pairs(quantized, mask, row, col)
        valid = first_mask & second_mask
        if not valid.any():
            raise ValueError("a ROI não tem pares vizinhos suficientes")
        matrix = np.zeros((levels, levels), dtype=float)
        np.add.at(matrix, (first[valid], second[valid]), 1)
        matrix += matrix.T
        matrix /= matrix.sum()
        contrasts.append(float((matrix * squared_difference).sum()))
        energies.append(float((matrix**2).sum()))
    result = {"contrast": float(np.mean(contrasts)), "energy": float(np.mean(energies))}
    for angle, contrast, energy in zip(ANGLES, contrasts, energies):
        result[f"contrast_{angle}deg"] = contrast
        result[f"energy_{angle}deg"] = energy
    return result
