"""Compara as gerações Dgen e Dsr sobre os mesmos cortes originais.

O Dsr não traz máscaras próprias; como as originais são byte a byte idênticas às
do Dgen, as máscaras do Dgen são reutilizadas após essa verificação explícita.
O relatório serve para triagem técnica e seleção de figuras, não para validação
clínica das imagens.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import laplace

from .glcm_features import glcm_contrast_energy
from .load_data import CasePaths, load_image
from .validate_cases import RejectedCase, validate_case


def original_paths(folder: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in folder.glob("*.tiff")
        if not path.name.endswith(("_generate.tiff", "_mask.tiff"))
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def display_limits(image: np.ndarray) -> tuple[float, float]:
    return tuple(float(value) for value in np.percentile(image, (1, 99)))


def sharpness(image: np.ndarray, mask: np.ndarray) -> float:
    """Variância do Laplaciano na ROI: indicador auxiliar de detalhe local."""
    return float(np.var(laplace(image)[mask]))


def save_contact_sheet(rows: list[dict], output: Path, title: str) -> None:
    fig, axes = plt.subplots(len(rows), 4, figsize=(15, 4 * len(rows)))
    axes = np.atleast_2d(axes)
    for axis_row, row in zip(axes, rows):
        original, dgen, dsr, mask = row["original_image"], row["dgen_image"], row["dsr_image"], row["mask"]
        low, high = display_limits(original)
        for axis, image, label in zip(axis_row[:3], (original, dgen, dsr), ("Original", "Dgen", "Dsr")):
            axis.imshow(image, cmap="gray", vmin=low, vmax=high)
            axis.contour(mask, levels=[0.5], colors="#00c980", linewidths=.7)
            axis.set_title(label)
            axis.axis("off")
        axis_row[3].imshow(np.abs(dsr - original) - np.abs(dgen - original), cmap="coolwarm")
        axis_row[3].set_title("Erro Dsr − erro Dgen\n(azul favorece Dsr)")
        axis_row[3].axis("off")
        axis_row[0].set_ylabel(
            f"{row['case_id'][-18:]}\nGLCM contrast: {row['dgen_contrast']:.2f} → {row['dsr_contrast']:.2f}\n"
            f"MAE: {row['dgen_mae']:.1f} → {row['dsr_mae']:.1f}",
            fontsize=8,
        )
    fig.suptitle(title, fontsize=16, y=.995)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dgen-dir", type=Path, required=True)
    parser.add_argument("--dsr-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, help="Pasta opcional para montagens visuais.")
    parser.add_argument("--levels", type=int, default=32)
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
        result = validate_case(paths)
        if isinstance(result, RejectedCase):
            rejected.append({"case_id": case, "reason": result.reason})
        else:
            valid.append(result)

    pixels = np.concatenate([case.original[case.lung_mask] for case in valid])
    clip_low, clip_high = (float(value) for value in np.percentile(pixels, (1, 99)))
    rows: list[dict] = []
    for case in valid:
        dsr = load_image(args.dsr_dir / f"{case.paths.case_id}_generate.tiff")
        if dsr.shape != case.original.shape:
            rejected.append({"case_id": case.paths.case_id, "reason": "Dsr com dimensão diferente"})
            continue
        dgen_metrics = glcm_contrast_energy(case.synthetic, case.lung_mask, clip_low, clip_high, args.levels)
        dsr_metrics = glcm_contrast_energy(dsr, case.lung_mask, clip_low, clip_high, args.levels)
        original_metrics = glcm_contrast_energy(case.original, case.lung_mask, clip_low, clip_high, args.levels)
        original_roi = case.original[case.lung_mask]
        dgen_roi, dsr_roi = case.synthetic[case.lung_mask], dsr[case.lung_mask]
        rows.append({
            "case_id": case.paths.case_id,
            "original_contrast": original_metrics["contrast"],
            "dgen_contrast": dgen_metrics["contrast"], "dsr_contrast": dsr_metrics["contrast"],
            "original_energy": original_metrics["energy"],
            "dgen_energy": dgen_metrics["energy"], "dsr_energy": dsr_metrics["energy"],
            "dgen_mae": float(np.mean(np.abs(dgen_roi - original_roi))),
            "dsr_mae": float(np.mean(np.abs(dsr_roi - original_roi))),
            "dgen_sharpness": sharpness(case.synthetic, case.lung_mask),
            "dsr_sharpness": sharpness(dsr, case.lung_mask),
            "dgen_contrast_error": abs(dgen_metrics["contrast"] - original_metrics["contrast"]),
            "dsr_contrast_error": abs(dsr_metrics["contrast"] - original_metrics["contrast"]),
            "original_image": case.original, "dgen_image": case.synthetic, "dsr_image": dsr, "mask": case.lung_mask,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    serializable = [{key: value for key, value in row.items() if key not in {"original_image", "dgen_image", "dsr_image", "mask"}} for row in rows]
    write_csv(args.output_dir / "metricas_por_caso.csv", serializable)
    write_csv(args.output_dir / "casos_invalidos.csv", rejected)
    summary = {
        "shared_pairs": len(shared), "valid_pairs": len(rows), "invalid_pairs": len(rejected),
        "clip_interval": [clip_low, clip_high], "levels": args.levels,
        "dgen_lower_mae": sum(row["dgen_mae"] < row["dsr_mae"] for row in rows),
        "dsr_lower_mae": sum(row["dsr_mae"] < row["dgen_mae"] for row in rows),
        "dgen_closer_contrast": sum(row["dgen_contrast_error"] < row["dsr_contrast_error"] for row in rows),
        "dsr_closer_contrast": sum(row["dsr_contrast_error"] < row["dgen_contrast_error"] for row in rows),
        "median_dgen_contrast": float(np.median([row["dgen_contrast"] for row in rows])),
        "median_dsr_contrast": float(np.median([row["dsr_contrast"] for row in rows])),
        "median_original_contrast": float(np.median([row["original_contrast"] for row in rows])),
        "median_dgen_mae": float(np.median([row["dgen_mae"] for row in rows])),
        "median_dsr_mae": float(np.median([row["dsr_mae"] for row in rows])),
    }
    import json
    (args.output_dir / "resumo.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.figures_dir:
        args.figures_dir.mkdir(parents=True, exist_ok=True)
        better_dsr = sorted(rows, key=lambda row: row["dsr_contrast_error"] - row["dgen_contrast_error"])[:5]
        worse_dsr = sorted(rows, key=lambda row: row["dsr_contrast_error"] - row["dgen_contrast_error"], reverse=True)[:5]
        save_contact_sheet(better_dsr, args.figures_dir / "comparacao_Dsr_melhor.png", "Casos em que Dsr preserva melhor o contraste GLCM")
        save_contact_sheet(worse_dsr, args.figures_dir / "comparacao_Dgen_melhor.png", "Casos em que Dgen preserva melhor o contraste GLCM")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
