"""Visualização de exemplo: mapa de energia wavelet no método multipatch vs imagem inteira.

Só para inspeção visual — não entra no pipeline de features. Reaproveita as
funções internas de `wavelet_features.py` (não alteradas) e a decomposição de
`wavelet_features_fullimage.py` para desenhar, lado a lado, como cada método
"vê" a textura espacialmente.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pywt

from .compare_patch_vs_fullimage import load_valid_cases
from .wavelet_features import MIN_TISSUE_FRACTION, PATCH_SIZE, PATCH_STRIDE, _patch_positions


def build_patch_energy_map(
    image: np.ndarray, mask: np.ndarray, clip_low: float, clip_high: float,
    wavelet: str, level: int, patch_size: int, patch_stride: int, min_tissue: float,
) -> np.ndarray:
    """Energia média do nível mais fino de cada patch, pintada de volta na grade da imagem."""
    clipped = np.clip(image, clip_low, clip_high)
    normalised = (clipped - clip_low) / (clip_high - clip_low)
    positions = _patch_positions(mask, patch_size, patch_stride, min_tissue)
    accum = np.zeros(mask.shape)
    count = np.zeros(mask.shape)
    for row, col in positions:
        coeffs = pywt.wavedec2(normalised[row:row + patch_size, col:col + patch_size], wavelet=wavelet, level=level)
        finest_bands = coeffs[-1]
        local_energy = float(np.mean([np.mean(band**2) for band in finest_bands]))
        accum[row:row + patch_size, col:col + patch_size] += local_energy
        count[row:row + patch_size, col:col + patch_size] += 1
    energy_map = np.full(mask.shape, np.nan)
    covered = count > 0
    energy_map[covered] = accum[covered] / count[covered]
    return energy_map


def build_fullimage_energy_map(
    image: np.ndarray, clip_low: float, clip_high: float, wavelet: str, level: int,
) -> np.ndarray:
    """Energia pixel-a-pixel do nível mais fino (LH/HL/HH), reamostrada de volta ao tamanho original."""
    clipped = np.clip(image, clip_low, clip_high)
    normalised = (clipped - clip_low) / (clip_high - clip_low)
    coeffs = pywt.wavedec2(normalised, wavelet=wavelet, level=level, mode="periodization")
    finest_bands = coeffs[-1]
    energy = np.mean([band**2 for band in finest_bands], axis=0)
    return np.kron(energy, np.ones((2, 2)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/patch_vs_fullimage/figures"))
    parser.add_argument("--mask-label", type=float, default=1.0)
    parser.add_argument("--erosion-radius", type=int, default=2)
    parser.add_argument("--wavelet", type=str, default="haar")
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--patch-stride", type=int, default=PATCH_STRIDE)
    parser.add_argument("--min-tissue", type=float, default=MIN_TISSUE_FRACTION)
    parser.add_argument("--case-index", type=int, default=0, help="Índice do caso válido a ilustrar.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    valid = load_valid_cases(args.data_dir, args.mask_label, args.erosion_radius)
    if not valid:
        raise RuntimeError("nenhum caso válido após o filtro de máscara")
    real_lung_pixels = np.concatenate([case.original[case.lung_mask] for case in valid])
    clip_low, clip_high = (float(value) for value in np.percentile(real_lung_pixels, (1, 99)))

    case = valid[args.case_index]
    mask = case.lung_mask

    patch_map = build_patch_energy_map(case.original, mask, clip_low, clip_high, args.wavelet, args.level, args.patch_size, args.patch_stride, args.min_tissue)
    full_map = build_fullimage_energy_map(case.original, clip_low, clip_high, args.wavelet, args.level)
    full_map_masked = np.where(mask, full_map, np.nan)

    display_low, display_high = np.percentile(case.original, (1, 99))
    energy_vmax = np.nanpercentile(np.concatenate([patch_map[~np.isnan(patch_map)], full_map_masked[~np.isnan(full_map_masked)]]), 99)

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    axes[0].imshow(case.original, cmap="gray", vmin=display_low, vmax=display_high)
    axes[0].contour(mask, colors="#F58518", linewidths=1)
    axes[0].set_title("CT original + máscara erodida")

    axes[1].imshow(case.original, cmap="gray", vmin=display_low, vmax=display_high)
    im1 = axes[1].imshow(patch_map, cmap="inferno", vmin=0, vmax=energy_vmax, alpha=.85)
    axes[1].set_title("Multipatch 16x16\n(energia do nível mais fino por patch)")
    fig.colorbar(im1, ax=axes[1], fraction=.046, pad=.04)

    axes[2].imshow(case.original, cmap="gray", vmin=display_low, vmax=display_high)
    im2 = axes[2].imshow(full_map_masked, cmap="inferno", vmin=0, vmax=energy_vmax, alpha=.85)
    axes[2].set_title("Imagem inteira + máscara\n(energia do nível mais fino, pixel a pixel)")
    fig.colorbar(im2, ax=axes[2], fraction=.046, pad=.04)

    diff = patch_map - full_map_masked
    diff_abs_max = np.nanpercentile(np.abs(diff), 99)
    im3 = axes[3].imshow(diff, cmap="coolwarm", vmin=-diff_abs_max, vmax=diff_abs_max)
    axes[3].set_title("Diferença\n(multipatch − imagem inteira)")
    fig.colorbar(im3, ax=axes[3], fraction=.046, pad=.04)

    for axis in axes:
        axis.axis("off")
    fig.suptitle(f"Exemplo de mapa de energia wavelet — caso {case.paths.case_id}")
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "wavelet_example_maps.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"Figura salva em {out_path.resolve()}")


if __name__ == "__main__":
    main()
