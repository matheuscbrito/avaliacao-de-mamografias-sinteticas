"""Compara as gerações Dgen e Dsr com as features wavelet (método de imagem inteira).

Espelha `glcm/src/compare_generations.py`: o Dsr não traz máscaras próprias, então
as máscaras do Dgen são reutilizadas depois de confirmar que as originais dos dois
conjuntos são idênticas. Os mesmos cortes passam pelo filtro de ROI do MVP, pela
referência construída só com originais aceitas e pela classificação inicial do
framework, uma vez para cada geração.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, wilcoxon

from .build_reference import FEATURES, build_reference
from .compare_images import compare_case
from .load_data import CasePaths, load_image
from .run_pipeline import write_csv
from .validate_cases import RejectedCase, validate_case
from .wavelet_features_fullimage import wavelet_energy_entropy_fullimage

GENERATIONS = ("dgen", "dsr")
BANDS = ("LH", "HL", "HH")
STATUSES = ("approved", "approved_with_observation", "manual_review", "reject")
COLORS = {"dgen": "#E45756", "dsr": "#4C78A8"}


def original_paths(folder: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in folder.glob("*.tiff")
        if not path.name.endswith(("_generate.tiff", "_mask.tiff"))
    }


def scale_keys(levels: int) -> list[tuple[int, str]]:
    """Pares (escala, chave) das sub-bandas; escala 1 é a mais fina.

    `wavelet_energy_entropy_fullimage` numera as chaves pela ordem do
    `wavedec2`, em que `L1` é o nível mais grosso.
    """
    return [(levels - depth + 1, f"L{depth}") for depth in range(1, levels + 1)]


def paired_test(dgen_errors: list[float], dsr_errors: list[float]) -> float | None:
    differences = np.array(dsr_errors) - np.array(dgen_errors)
    if not np.any(differences):
        return None
    return float(wilcoxon(dsr_errors, dgen_errors).pvalue)


def save_scatter(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, feature, title in zip(axes, FEATURES, ("Energia wavelet", "Entropia wavelet")):
        x = [row[f"original_{feature}"] for row in rows]
        values = list(x)
        for generation in GENERATIONS:
            y = [row[f"{generation}_{feature}"] for row in rows]
            values += y
            axis.scatter(x, y, color=COLORS[generation], alpha=.8, label=generation.capitalize())
        lower, upper = min(values), max(values)
        axis.plot([lower, upper], [lower, upper], "--", color="#888888", label="sintética = original")
        axis.set(title=title, xlabel="Original", ylabel="Sintética")
        axis.legend(fontsize=8)
    fig.suptitle("Cada ponto é um corte; mais perto da diagonal = textura mais preservada")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def save_subband_ratios(rows: list[dict], levels: int, output: Path) -> None:
    labels, data, colors = [], [], []
    for scale, _ in sorted(scale_keys(levels)):
        for band in BANDS:
            for generation in GENERATIONS:
                labels.append(f"{band} e{scale}\n{generation.capitalize()}")
                data.append([row[f"log2_energy_ratio_{generation}_s{scale}_{band}"] for row in rows])
                colors.append(COLORS[generation])
    fig, axis = plt.subplots(figsize=(13, 4.8))
    boxes = axis.boxplot(data, patch_artist=True, widths=.6)
    for patch, color in zip(boxes["boxes"], colors):
        patch.set(facecolor=color, alpha=.6)
    axis.axhline(0, color="#888888", linestyle="--")
    axis.set_xticks(range(1, len(labels) + 1), labels, fontsize=7)
    axis.set(ylabel="log2(energia sintética / original)", title="Energia por sub-banda (e1 = escala mais fina); abaixo de 0 = detalhe perdido")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dgen-dir", type=Path, required=True)
    parser.add_argument("--dsr-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wavelet", type=str, default="haar")
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--min-tissue", type=float, default=0.9)
    parser.add_argument("--erosion-radius", type=int, default=2)
    parser.add_argument("--pair-iqr-threshold", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dgen_originals, dsr_originals = original_paths(args.dgen_dir), original_paths(args.dsr_dir)
    shared = sorted(dgen_originals.keys() & dsr_originals.keys())
    if len(shared) != len(dgen_originals) or len(shared) != len(dsr_originals):
        raise RuntimeError("Dgen e Dsr não têm o mesmo conjunto de originais")
    if any(not np.array_equal(load_image(dgen_originals[case]), load_image(dsr_originals[case])) for case in shared):
        raise RuntimeError("as originais Dgen e Dsr diferem; não é seguro reutilizar as máscaras")

    valid, rejected = [], []
    for case in shared:
        paths = CasePaths(
            case_id=case,
            original=dgen_originals[case],
            synthetic=args.dgen_dir / f"{case}_generate.tiff",
            mask=args.dgen_dir / f"{case}_mask.tiff",
        )
        result = validate_case(paths, erosion_radius=args.erosion_radius)
        if isinstance(result, RejectedCase):
            rejected.append({"case_id": case, "reason": result.reason})
            continue
        dsr = load_image(args.dsr_dir / f"{case}_generate.tiff")
        if dsr.shape != result.original.shape:
            rejected.append({"case_id": case, "reason": "Dsr com dimensão diferente"})
            continue
        valid.append((result, dsr))

    pixels = np.concatenate([case.original[case.lung_mask] for case, _ in valid])
    clip_low, clip_high = (float(value) for value in np.percentile(pixels, (1, 99)))

    def features(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        return wavelet_energy_entropy_fullimage(image, mask, clip_low, clip_high, args.wavelet, args.level, args.min_tissue)

    metrics = [
        (case.paths.case_id, features(case.original, case.lung_mask), {"dgen": features(case.synthetic, case.lung_mask), "dsr": features(dsr, case.lung_mask)})
        for case, dsr in valid
    ]
    reference = build_reference([original for _, original, _ in metrics])
    levels = int(metrics[0][1]["levels_used"])

    rows: list[dict] = []
    for case_id, original, generated in metrics:
        row: dict = {"case_id": case_id}
        for feature in FEATURES:
            row[f"original_{feature}"] = original[feature]
            for generation in GENERATIONS:
                row[f"{generation}_{feature}"] = generated[generation][feature]
                row[f"{generation}_{feature}_error"] = abs(generated[generation][feature] - original[feature])
        for generation in GENERATIONS:
            decision = compare_case(case_id, original, generated[generation], reference, args.pair_iqr_threshold)
            row[f"{generation}_status"] = decision["status"]
            for feature in FEATURES:
                row[f"{generation}_pair_distance_{feature}_iqr"] = decision[f"pair_distance_{feature}_iqr"]
            for scale, key in scale_keys(levels):
                for band in BANDS:
                    ratio = generated[generation][f"energy_{key}_{band}"] / max(original[f"energy_{key}_{band}"], 1e-12)
                    row[f"log2_energy_ratio_{generation}_s{scale}_{band}"] = float(np.log2(max(ratio, 1e-12)))
        rows.append(row)

    summary: dict = {
        "shared_pairs": len(shared), "valid_pairs": len(rows), "invalid_pairs": len(rejected),
        "clip_interval": [clip_low, clip_high], "wavelet": args.wavelet, "level": levels,
        "min_tissue": args.min_tissue, "pair_iqr_threshold": args.pair_iqr_threshold,
    }
    for feature in FEATURES:
        dgen_errors = [row[f"dgen_{feature}_error"] for row in rows]
        dsr_errors = [row[f"dsr_{feature}_error"] for row in rows]
        summary[feature] = {
            "median_original": float(np.median([row[f"original_{feature}"] for row in rows])),
            "median_dgen": float(np.median([row[f"dgen_{feature}"] for row in rows])),
            "median_dsr": float(np.median([row[f"dsr_{feature}"] for row in rows])),
            "median_dgen_error": float(np.median(dgen_errors)),
            "median_dsr_error": float(np.median(dsr_errors)),
            "median_dgen_pair_distance_iqr": float(np.median([row[f"dgen_pair_distance_{feature}_iqr"] for row in rows])),
            "median_dsr_pair_distance_iqr": float(np.median([row[f"dsr_pair_distance_{feature}_iqr"] for row in rows])),
            "dsr_closer": sum(dsr < dgen for dgen, dsr in zip(dgen_errors, dsr_errors)),
            "dgen_closer": sum(dgen < dsr for dgen, dsr in zip(dgen_errors, dsr_errors)),
            "wilcoxon_p": paired_test(dgen_errors, dsr_errors),
            **{f"spearman_original_{generation}": float(spearmanr([row[f"original_{feature}"] for row in rows], [row[f"{generation}_{feature}"] for row in rows])[0]) for generation in GENERATIONS},
        }
    summary["status_counts"] = {generation: {status: sum(row[f"{generation}_status"] == status for row in rows) for status in STATUSES} for generation in GENERATIONS}
    summary["median_log2_energy_ratio"] = {
        f"s{scale}_{band}": {generation: float(np.median([row[f"log2_energy_ratio_{generation}_s{scale}_{band}"] for row in rows])) for generation in GENERATIONS}
        for scale, _ in sorted(scale_keys(levels)) for band in BANDS
    }

    figures = args.output_dir / "figuras"
    figures.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "metricas_por_caso.csv", rows)
    write_csv(args.output_dir / "casos_invalidos.csv", rejected)
    (args.output_dir / "resumo.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_scatter(rows, figures / "original_vs_sinteticas.png")
    save_subband_ratios(rows, levels, figures / "energia_por_subbanda.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
