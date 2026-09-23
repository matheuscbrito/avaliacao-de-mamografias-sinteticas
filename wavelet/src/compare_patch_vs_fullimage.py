"""Experimento: multipatch 16x16 (método atual) vs wavelet na imagem inteira + seleção por máscara.

Não altera `wavelet_features.py`. Roda as duas abordagens sobre os mesmos casos
válidos e grava CSVs, figuras e um relatório com achados empíricos sobre
cobertura, distribuição das features e efeitos de resolução/borda.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pywt

from .load_data import discover_cases
from .run_pipeline import write_csv
from .validate_cases import RejectedCase, ValidCase, validate_case
from .wavelet_features import wavelet_energy_entropy
from .wavelet_features_fullimage import wavelet_energy_entropy_fullimage

SUBBAND_KEYS = ("L1_LH", "L1_HL", "L1_HH", "L2_LH", "L2_HL", "L2_HH")


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    q1, q3 = np.percentile(array, (25, 75))
    return {
        "n": int(array.size), "mean": float(array.mean()), "median": float(np.median(array)),
        "std": float(array.std(ddof=0)), "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
        "min": float(array.min()), "max": float(array.max()),
    }


def pearson(x: list[float], y: list[float]) -> float:
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if xa.std() == 0 or ya.std() == 0:
        return float("nan")
    return float(np.corrcoef(xa, ya)[0, 1])


def load_valid_cases(data_dir: Path, mask_label: float, erosion_radius: int) -> list[ValidCase]:
    candidates = discover_cases(data_dir)
    valid = []
    for paths in candidates:
        result = validate_case(paths, mask_label=mask_label, erosion_radius=erosion_radius)
        if not isinstance(result, RejectedCase):
            valid.append(result)
    return valid


def compute_feature_rows(
    valid: list[ValidCase], clip_low: float, clip_high: float, wavelet: str, level: int,
    patch_size: int, patch_stride: int, min_tissue: float,
) -> list[dict]:
    rows = []
    for case in valid:
        for image_type, image in (("original", case.original), ("synthetic", case.synthetic)):
            patch = wavelet_energy_entropy(image, case.lung_mask, clip_low, clip_high, wavelet, level, patch_size, patch_stride, min_tissue)
            full = wavelet_energy_entropy_fullimage(image, case.lung_mask, clip_low, clip_high, wavelet, level, min_tissue)
            row = {
                "case_id": case.paths.case_id, "image_type": image_type,
                "patch_energy": patch["energy"], "patch_entropy": patch["entropy"],
                "patch_tissue_coverage": patch["tissue_coverage"], "patch_count": patch["patch_count"],
                "patch_levels_used": patch["levels_used"],
                "full_energy": full["energy"], "full_entropy": full["entropy"],
                "full_tissue_coverage": full["tissue_coverage"], "full_levels_used": full["levels_used"],
            }
            for key in SUBBAND_KEYS:
                row[f"patch_energy_{key}"] = patch.get(f"energy_{key}")
                row[f"patch_entropy_{key}"] = patch.get(f"entropy_{key}")
                row[f"full_energy_{key}"] = full.get(f"energy_{key}")
                row[f"full_entropy_{key}"] = full.get(f"entropy_{key}")
                row[f"full_n_coeffs_{key}"] = full.get(f"n_coeffs_{key}")
            rows.append(row)
    return rows


def build_summary(rows: list[dict]) -> list[dict]:
    summary = []
    for image_type in ("original", "synthetic", "all"):
        subset = rows if image_type == "all" else [r for r in rows if r["image_type"] == image_type]
        for feature in ("energy", "entropy", "tissue_coverage"):
            for method in ("patch", "full"):
                values = [r[f"{method}_{feature}"] for r in subset]
                summary.append({"image_type": image_type, "method": method, "feature": feature, **summarize(values)})
    return summary


def build_paired_comparison(rows: list[dict]) -> tuple[list[dict], dict]:
    paired = []
    for r in rows:
        entry = {"case_id": r["case_id"], "image_type": r["image_type"]}
        for feature in ("energy", "entropy", "tissue_coverage"):
            patch_value, full_value = r[f"patch_{feature}"], r[f"full_{feature}"]
            entry[f"{feature}_patch"] = patch_value
            entry[f"{feature}_full"] = full_value
            entry[f"{feature}_delta"] = full_value - patch_value
            entry[f"{feature}_relative_delta"] = (full_value - patch_value) / max(abs(patch_value), 1e-12)
        paired.append(entry)
    correlations = {
        feature: pearson([r[f"{feature}_patch"] for r in paired], [r[f"{feature}_full"] for r in paired])
        for feature in ("energy", "entropy", "tissue_coverage")
    }
    return paired, correlations


def resolution_report(patch_size: int, level: int, wavelet: str, image_shape: tuple[int, int]) -> dict:
    """Compara formas de sub-banda e crescimento por padding entre janela 16x16 e imagem inteira."""
    max_level_patch = pywt.dwtn_max_level((patch_size, patch_size), wavelet)
    max_level_full = pywt.dwtn_max_level(image_shape, wavelet)
    used_level = min(level, max_level_patch)

    def band_shapes(shape: tuple[int, int], mode: str) -> list[tuple[int, int]]:
        dummy = np.zeros(shape)
        coeffs = pywt.wavedec2(dummy, wavelet=wavelet, level=used_level, mode=mode)
        return [bands[0].shape for bands in coeffs[1:]]

    report = {
        "wavelet": wavelet, "requested_level": level, "used_level": used_level,
        "patch_size": patch_size, "image_shape": list(image_shape),
        "max_level_for_patch": max_level_patch, "max_level_for_full_image": max_level_full,
        "filter_length": pywt.Wavelet(wavelet).dec_len,
        "patch_subband_shapes_periodization": band_shapes((patch_size, patch_size), "periodization"),
        "patch_subband_shapes_symmetric": band_shapes((patch_size, patch_size), "symmetric"),
        "full_image_subband_shapes_periodization": band_shapes(image_shape, "periodization"),
        "full_image_subband_shapes_symmetric": band_shapes(image_shape, "symmetric"),
        "note": (
            "Para 'haar' (filtro de comprimento 2) os dois modos coincidem: não há "
            "crescimento por padding porque o suporte do filtro não ultrapassa o bloco "
            "diádico. Para wavelets mais longas (ex.: db4, comprimento 8) o modo "
            "'symmetric' (padrão do PyWavelets) produz sub-bandas maiores que N/2^nível "
            "e mistura pixels fora do bloco nominal nos coeficientes de borda — por "
            "isso este experimento usa 'periodization' na abordagem de imagem inteira."
        ),
    }
    return report


def boundary_leakage_probe(valid: list[ValidCase], clip_low: float, clip_high: float, wavelet: str, level: int, min_tissue: float, sample_size: int) -> dict:
    """Mede, para uma wavelet com suporte maior que 2 (ex. db4), quanto os modos de borda mudam a energia/entropia."""
    sample = valid[:sample_size]
    periodization, symmetric = [], []
    for case in sample:
        periodization.append(wavelet_energy_entropy_fullimage(case.original, case.lung_mask, clip_low, clip_high, wavelet, level, min_tissue, dwt_mode="periodization"))
        symmetric.append(wavelet_energy_entropy_fullimage(case.original, case.lung_mask, clip_low, clip_high, wavelet, level, min_tissue, dwt_mode="symmetric"))
    return {
        "wavelet": wavelet, "n_cases": len(sample),
        "energy_periodization_mean": float(np.mean([r["energy"] for r in periodization])),
        "energy_symmetric_mean": float(np.mean([r["energy"] for r in symmetric])),
        "entropy_periodization_mean": float(np.mean([r["entropy"] for r in periodization])),
        "entropy_symmetric_mean": float(np.mean([r["entropy"] for r in symmetric])),
        "tissue_coverage_periodization_mean": float(np.mean([r["tissue_coverage"] for r in periodization])),
        "tissue_coverage_symmetric_mean": float(np.mean([r["tissue_coverage"] for r in symmetric])),
    }


def save_figures(rows: list[dict], figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    originals = [r for r in rows if r["image_type"] == "original"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, feature, title in zip(axes, ("energy", "entropy"), ("Energia", "Entropia")):
        patch_values = [r[f"patch_{feature}"] for r in rows]
        full_values = [r[f"full_{feature}"] for r in rows]
        axis.hist(patch_values, bins="auto", alpha=.6, label="multipatch 16x16", color="#4C78A8")
        axis.hist(full_values, bins="auto", alpha=.6, label="imagem inteira + máscara", color="#F58518")
        axis.set(title=f"Distribuição de {title.lower()}", xlabel="Valor", ylabel="Contagem")
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "distributions_patch_vs_fullimage.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis, feature, title in zip(axes, ("energy", "entropy"), ("Energia", "Entropia")):
        x = [r[f"patch_{feature}"] for r in rows]
        y = [r[f"full_{feature}"] for r in rows]
        colors = ["#4C78A8" if r["image_type"] == "original" else "#F58518" for r in rows]
        axis.scatter(x, y, c=colors, alpha=.8)
        lower, upper = min(x + y), max(x + y)
        axis.plot([lower, upper], [lower, upper], "--", color="#888888", label="patch = imagem inteira")
        corr = pearson(x, y)
        axis.set(title=f"{title}: patch vs imagem inteira (r={corr:.2f})", xlabel="Multipatch 16x16", ylabel="Imagem inteira + máscara")
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "scatter_patch_vs_fullimage.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis, feature, title in zip(axes, ("energy", "entropy"), ("Energia", "Entropia")):
        x = [r[f"patch_{feature}"] for r in rows]
        y = [r[f"full_{feature}"] for r in rows]
        mean = [(a + b) / 2 for a, b in zip(x, y)]
        diff = [b - a for a, b in zip(x, y)]
        bias, std = float(np.mean(diff)), float(np.std(diff))
        axis.scatter(mean, diff, color="#54A24B", alpha=.8)
        axis.axhline(bias, color="#E45756", label=f"viés={bias:.4f}")
        axis.axhline(bias + 1.96 * std, color="#B279A2", linestyle="--", label="±1.96 dp")
        axis.axhline(bias - 1.96 * std, color="#B279A2", linestyle="--")
        axis.set(title=f"Bland-Altman: {title.lower()}", xlabel="Média dos métodos", ylabel="Imagem inteira − patch")
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "bland_altman.png", dpi=170)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 4.5))
    axis.scatter([r["patch_tissue_coverage"] for r in rows], [r["full_tissue_coverage"] for r in rows], color="#4C78A8", alpha=.8)
    axis.plot([0, 1], [0, 1], "--", color="#888888")
    axis.set(title="Cobertura da ROI: patch vs imagem inteira", xlabel="Multipatch 16x16", ylabel="Imagem inteira + máscara", xlim=(0, 1.02), ylim=(0, 1.02))
    fig.tight_layout()
    fig.savefig(figures / "tissue_coverage_comparison.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, method, title in zip(axes, ("patch", "full"), ("Multipatch 16x16", "Imagem inteira + máscara")):
        data = [[r[f"{method}_energy_{key}"] for r in originals] for key in SUBBAND_KEYS]
        axis.boxplot(data, tick_labels=SUBBAND_KEYS)
        axis.set(title=f"Energia por sub-banda — {title}", ylabel="Energia")
        axis.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(figures / "subband_energy_by_method.png", dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/patch_vs_fullimage"))
    parser.add_argument("--mask-label", type=float, default=1.0)
    parser.add_argument("--erosion-radius", type=int, default=2)
    parser.add_argument("--wavelet", type=str, default="haar")
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=4)
    parser.add_argument("--min-tissue", type=float, default=0.9)
    parser.add_argument("--boundary-probe-wavelet", type=str, default="db4", help="Wavelet de suporte longo usada só para medir o efeito de padding/borda.")
    parser.add_argument("--boundary-probe-n", type=int, default=10, help="Quantos casos usar na sonda de borda (mantém o experimento rápido).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output, figures = args.output_dir, args.output_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)

    valid = load_valid_cases(args.data_dir, args.mask_label, args.erosion_radius)
    if not valid:
        raise RuntimeError("nenhum caso válido após o filtro de máscara")

    real_lung_pixels = np.concatenate([case.original[case.lung_mask] for case in valid])
    clip_low, clip_high = (float(value) for value in np.percentile(real_lung_pixels, (1, 99)))

    rows = compute_feature_rows(valid, clip_low, clip_high, args.wavelet, args.level, args.patch_size, args.patch_stride, args.min_tissue)
    write_csv(output / "comparison_features.csv", rows)

    summary = build_summary(rows)
    write_csv(output / "summary_stats.csv", summary)

    paired, correlations = build_paired_comparison(rows)
    write_csv(output / "paired_comparison.csv", paired)

    res_report = resolution_report(args.patch_size, args.level, args.wavelet, valid[0].original.shape)
    boundary_report = boundary_leakage_probe(valid, clip_low, clip_high, args.boundary_probe_wavelet, args.level, args.min_tissue, args.boundary_probe_n)

    with (output / "resolution_and_boundary_report.json").open("w", encoding="utf-8") as file:
        json.dump({
            "n_valid_cases": len(valid), "n_rows": len(rows),
            "clip_low": clip_low, "clip_high": clip_high,
            "wavelet": args.wavelet, "level": args.level,
            "patch_size": args.patch_size, "patch_stride": args.patch_stride, "min_tissue": args.min_tissue,
            "correlations_patch_vs_full": correlations,
            "resolution_report": res_report,
            "boundary_leakage_probe": boundary_report,
        }, file, indent=2)

    save_figures(rows, figures)
    print(f"Experimento concluído: {len(valid)} casos válidos, {len(rows)} linhas (original+sintética). Resultados: {output.resolve()}")
    print(f"Correlação (Pearson) patch vs imagem inteira: {correlations}")


if __name__ == "__main__":
    main()
