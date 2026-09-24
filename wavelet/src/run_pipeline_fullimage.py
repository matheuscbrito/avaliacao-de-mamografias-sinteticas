"""Executa o MVP wavelet com o método adotado (imagem inteira + máscara): seleção, referência, comparação e figuras.

Espelha `run_pipeline.py` (que continua usando o multipatch e não foi alterado),
trocando só a extração de features por `wavelet_energy_entropy_fullimage`. Saídas
em pastas próprias para não sobrescrever os resultados do método antigo.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .build_reference import FEATURES, build_reference, reference_rows
from .compare_images import compare_case
from .load_data import discover_cases
from .run_pipeline import display_limits, write_csv
from .validate_cases import RejectedCase, ValidCase, validate_case
from .wavelet_features_fullimage import wavelet_energy_entropy_fullimage


def save_figures(valid: list[ValidCase], originals: list[dict], synthetics: list[dict], reference: dict, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, feature, title in zip(axes, FEATURES, ("Energia wavelet (imagem inteira)", "Entropia wavelet (imagem inteira)")):
        values = np.array([row[feature] for row in originals])
        stats = reference[feature]
        axis.hist(values, bins="auto", color="#4C78A8", edgecolor="white")
        axis.axvline(stats["mean"], color="#E45756", label="média")
        axis.axvline(stats["median"], color="#54A24B", label="mediana")
        axis.axvline(stats["tukey_lower"], color="#B279A2", linestyle="--", label="limites esperados")
        axis.axvline(stats["tukey_upper"], color="#B279A2", linestyle="--")
        axis.set(title=title, xlabel="Valor da feature", ylabel="Número de originais")
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "reference_distributions.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis, feature, title in zip(axes, FEATURES, ("Energia", "Entropia")):
        x, y = [row[feature] for row in originals], [row[feature] for row in synthetics]
        axis.scatter(x, y, color="#4C78A8", alpha=.8)
        lower, upper = min(x + y), max(x + y)
        axis.plot([lower, upper], [lower, upper], "--", color="#888888", label="sintética = original")
        axis.set(title=title, xlabel="Original", ylabel="Sintética")
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "original_vs_synthetic.png", dpi=170)
    plt.close(fig)

    if valid:
        sample = valid[0]
        fig, axes = plt.subplots(1, 3, figsize=(11, 4))
        low, high = display_limits(sample.original)
        axes[0].imshow(sample.original, cmap="gray", vmin=low, vmax=high)
        axes[0].set_title("Original")
        axes[1].imshow(sample.synthetic, cmap="gray", vmin=low, vmax=high)
        axes[1].set_title("Sintética")
        axes[2].imshow(sample.lung_mask, cmap="gray")
        axes[2].set_title("Máscara erodida usada")
        for axis in axes:
            axis.axis("off")
        fig.suptitle(f"Exemplo de caso aceito: {sample.paths.case_id}")
        fig.tight_layout()
        fig.savefig(figures / "example_case.png", dpi=170)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Pasta plana de TIFFs ou pasta contendo original/, synthetic/ e masks/.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fullimage"), help="Pasta dos CSVs e figuras.")
    parser.add_argument("--mask-label", type=float, default=1.0, help="Valor da máscara que representa pulmão.")
    parser.add_argument("--erosion-radius", type=int, default=2, help="Erosão da máscara em pixels.")
    parser.add_argument("--wavelet", type=str, default="haar", help="Família wavelet usada na decomposição (nomes do PyWavelets).")
    parser.add_argument("--level", type=int, default=2, help="Número de níveis de decomposição wavelet.")
    parser.add_argument("--min-tissue", type=float, default=0.9, help="Fração mínima de pulmão exigida para aceitar um coeficiente.")
    parser.add_argument("--pair-iqr-threshold", type=float, default=1.0, help="Distância máxima do par, medida em IQRs da referência, para considerar o par próximo.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output, figures = args.output_dir, args.output_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    candidates = discover_cases(args.data_dir)
    if not candidates:
        raise RuntimeError("nenhuma imagem original encontrada")

    valid: list[ValidCase] = []
    validation_rows: list[dict] = []
    for paths in candidates:
        result = validate_case(paths, mask_label=args.mask_label, erosion_radius=args.erosion_radius)
        if isinstance(result, RejectedCase):
            validation_rows.append({"case_id": paths.case_id, "status": "rejected", "reason": result.reason, "original_path": str(paths.original), "synthetic_path": str(paths.synthetic), "mask_path": str(paths.mask)})
        else:
            valid.append(result)
            validation_rows.append({"case_id": paths.case_id, "status": "accepted", "reason": "máscara e par válidos", "original_path": str(paths.original), "synthetic_path": str(paths.synthetic), "mask_path": str(paths.mask), "lung_area_fraction": result.area_fraction, "component_count": result.component_count, "roi_pixels_after_erosion": int(result.lung_mask.sum())})
    if not valid:
        raise RuntimeError("nenhum caso válido após o filtro de máscara")
    write_csv(output / "case_validation.csv", validation_rows)

    real_lung_pixels = np.concatenate([case.original[case.lung_mask] for case in valid])
    clip_low, clip_high = (float(value) for value in np.percentile(real_lung_pixels, (1, 99)))
    originals, synthetics = [], []
    for case in valid:
        original_metrics = wavelet_energy_entropy_fullimage(case.original, case.lung_mask, clip_low, clip_high, args.wavelet, args.level, args.min_tissue)
        synthetic_metrics = wavelet_energy_entropy_fullimage(case.synthetic, case.lung_mask, clip_low, clip_high, args.wavelet, args.level, args.min_tissue)
        originals.append({"case_id": case.paths.case_id, "image_type": "original", **original_metrics})
        synthetics.append({"case_id": case.paths.case_id, "image_type": "synthetic", **synthetic_metrics})
    feature_rows = originals + synthetics
    write_csv(output / "features_wavelet.csv", feature_rows)

    reference = build_reference(originals)
    write_csv(output / "reference_originals.csv", reference_rows(reference))
    with (output / "reference_originals.json").open("w", encoding="utf-8") as file:
        json.dump({"n_valid_originals": len(originals), "clip_low": clip_low, "clip_high": clip_high, "wavelet": args.wavelet, "level": args.level, "min_tissue": args.min_tissue, "erosion_radius": args.erosion_radius, "method": "fullimage", "reference": reference}, file, indent=2)

    original_by_id = {row["case_id"]: row for row in originals}
    comparisons = [compare_case(row["case_id"], original_by_id[row["case_id"]], row, reference, args.pair_iqr_threshold) for row in synthetics]
    write_csv(output / "comparison_results.csv", comparisons)
    save_figures(valid, originals, synthetics, reference, figures)
    print(f"Pipeline (imagem inteira) concluída: {len(valid)} aceitos de {len(candidates)} candidatos. Resultados: {output.resolve()}")


if __name__ == "__main__":
    main()
