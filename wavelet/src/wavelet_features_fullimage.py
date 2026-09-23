"""Wavelet na imagem inteira: decompõe uma vez e seleciona coeficientes pela máscara erodida.

Método alternativo ao multipatch de `wavelet_features.py` (não alterado). Em vez de
recortar janelas 16x16 e decompor cada uma, decompõe a imagem inteira uma única vez
e usa a máscara (reduzida ao tamanho de cada sub-banda) para escolher quais
coeficientes pertencem à ROI pulmonar antes de medir energia/entropia.
"""
from __future__ import annotations

import numpy as np
import pywt

MIN_TISSUE_FRACTION = 0.9


def _block_mean(mask: np.ndarray, block: int) -> np.ndarray:
    """Fração de tecido pulmonar em cada bloco `block`x`block` não sobreposto.

    Corresponde à região do domínio espacial que colapsa em um único
    coeficiente de uma sub-banda `block`x menor que a imagem original.
    """
    height, width = mask.shape
    trimmed = mask[: height - height % block, : width - width % block]
    reshaped = trimmed.reshape(trimmed.shape[0] // block, block, trimmed.shape[1] // block, block)
    return reshaped.astype(np.float64).mean(axis=(1, 3))


def _energy_entropy_from_values(values: np.ndarray) -> tuple[float, float]:
    """Mesma definição de `wavelet_features._subband_energy_entropy`, mas para um vetor já selecionado."""
    energy = float(np.mean(values**2))
    magnitude = np.abs(values)
    total = magnitude.sum()
    if total <= 0 or values.size < 2:
        return energy, 0.0
    probability = magnitude[magnitude > 0] / total
    entropy = float(-(probability * np.log2(probability)).sum())
    return energy, entropy / np.log2(values.size)


def wavelet_energy_entropy_fullimage(
    image: np.ndarray,
    mask: np.ndarray,
    clip_low: float,
    clip_high: float,
    wavelet: str = "haar",
    level: int = 2,
    min_tissue: float = MIN_TISSUE_FRACTION,
    dwt_mode: str = "periodization",
) -> dict[str, float]:
    """Decompõe a imagem inteira e mede energia/entropia apenas nos coeficientes cobertos pela ROI.

    Usa `mode="periodization"` por padrão: é o único modo do PyWavelets em que cada
    sub-banda do nível L mede exatamente altura/2^L por largura/2^L, sem o
    acréscimo de amostras de borda que os modos 'symmetric'/'reflect'/'zero'
    introduzem para wavelets com suporte maior que 2 (ex.: db4). Isso mantém a
    correspondência espacial coeficiente↔pixel simples e determinística, o que é
    exigido para reduzir a máscara ao tamanho de cada sub-banda com `_block_mean`.
    """
    if clip_high <= clip_low:
        raise ValueError("faixa de clipping inválida")
    if not mask.any():
        raise ValueError("a ROI não tem pixels marcados")
    max_level = pywt.dwtn_max_level(image.shape, wavelet)
    if max_level < 1:
        raise ValueError(f"imagem {image.shape} é pequena demais para a wavelet {wavelet}")
    used_level = min(level, max_level)

    clipped = np.clip(image, clip_low, clip_high)
    normalised = (clipped - clip_low) / (clip_high - clip_low)
    coeffs = pywt.wavedec2(normalised, wavelet=wavelet, level=used_level, mode=dwt_mode)

    energies: dict[str, list[float]] = {}
    entropies: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    selected_pixels_total = 0
    band_pixels_total = 0
    total_lung_pixels = int(mask.sum())
    for depth, bands in enumerate(coeffs[1:], start=1):
        true_pywt_level = used_level - depth + 1
        block = 2**true_pywt_level
        band_shape = bands[0].shape
        mask_fraction = _block_mean(mask, block)
        # A imagem pode não ser múltiplo exato do bloco nas bordas; alinhamos ao
        # tamanho real da sub-banda (pode ser 1px menor em modos não periódicos).
        mask_fraction = mask_fraction[: band_shape[0], : band_shape[1]]
        selection = mask_fraction >= min_tissue
        selected_pixels_total += int(selection.sum()) * block * block
        band_pixels_total += int(mask_fraction.size) * block * block
        key = f"L{depth}"
        for name, band in zip(("LH", "HL", "HH"), bands):
            band = band[: mask_fraction.shape[0], : mask_fraction.shape[1]]
            values = band[selection]
            counts[f"{key}_{name}"] = int(values.size)
            if values.size == 0:
                energies.setdefault(f"{key}_{name}", []).append(0.0)
                entropies.setdefault(f"{key}_{name}", []).append(0.0)
                continue
            energy, entropy = _energy_entropy_from_values(values)
            energies.setdefault(f"{key}_{name}", []).append(energy)
            entropies.setdefault(f"{key}_{name}", []).append(entropy)

    result: dict[str, float] = {}
    for key in energies:
        result[f"energy_{key}"] = float(np.mean(energies[key]))
        result[f"entropy_{key}"] = float(np.mean(entropies[key]))
        result[f"n_coeffs_{key}"] = float(counts[key])
    result["energy"] = float(np.mean([value for values in energies.values() for value in values]))
    result["entropy"] = float(np.mean([value for values in entropies.values() for value in values]))
    result["levels_used"] = float(used_level)
    result["selected_coeff_pixels_equiv"] = float(selected_pixels_total)
    result["band_coeff_pixels_equiv"] = float(band_pixels_total)
    # cobertura análoga a `tissue_coverage` do método patch: fração dos pixels
    # pulmonares reais que caem dentro de blocos selecionados na sub-banda mais
    # fina (nível 1 real, última tupla de `coeffs`, i.e. depth == used_level).
    finest_block = 2**1
    finest_mask_fraction = _block_mean(mask, finest_block)
    finest_selection = finest_mask_fraction >= min_tissue
    covered = np.zeros(mask.shape, dtype=bool)
    trimmed_h = finest_selection.shape[0] * finest_block
    trimmed_w = finest_selection.shape[1] * finest_block
    covered[:trimmed_h, :trimmed_w] = np.repeat(np.repeat(finest_selection, finest_block, axis=0), finest_block, axis=1)
    result["tissue_coverage"] = float((covered & mask).sum() / total_lung_pixels) if total_lung_pixels else 0.0
    return result
