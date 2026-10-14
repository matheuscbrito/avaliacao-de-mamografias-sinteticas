"""Executa o protocolo GLCM a partir de um manifesto explícito de pares."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.morphology import disk, erosion

from .core import calibrate_robust_scale, extract_glcm


def _read_image(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path), dtype=float)
    if image.ndim != 2:
        raise ValueError(f"{path}: a imagem deve ser 2D; recebido {image.shape}")
    if not np.isfinite(image).all():
        raise ValueError(f"{path}: contém valores não finitos")
    return image


def _resolve(value: str, manifest_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_dir / path


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    required = {"case_id", "real_path", "synthetic_path", "mask_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("manifesto precisa das colunas: case_id, real_path, synthetic_path, mask_path")
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mask-label", type=float, default=1.0)
    parser.add_argument("--erosion-radius", type=int, default=2)
    parser.add_argument("--tukey-k", type=float, default=3.0)
    parser.add_argument("--levels", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.erosion_radius < 0:
        raise ValueError("erosion-radius deve ser não negativo")
    manifest = _read_manifest(args.manifest)
    accepted, rejected = [], []
    for row in manifest:
        try:
            real = _read_image(_resolve(row["real_path"], args.manifest.parent))
            synthetic = _read_image(_resolve(row["synthetic_path"], args.manifest.parent))
            raw_mask = _read_image(_resolve(row["mask_path"], args.manifest.parent))
            if real.shape != synthetic.shape or real.shape != raw_mask.shape:
                raise ValueError("real, sintética e máscara têm dimensões diferentes")
            roi = np.isclose(raw_mask, args.mask_label)
            if args.erosion_radius:
                roi = erosion(roi, footprint=disk(args.erosion_radius))
            if not roi.any():
                raise ValueError("ROI vazia após erosão")
            accepted.append({"case_id": row["case_id"], "real": real, "synthetic": synthetic, "roi": roi})
        except Exception as error:  # preserva a razão da exclusão na saída
            rejected.append({"case_id": row.get("case_id", ""), "reason": str(error)})
    if not accepted:
        raise RuntimeError("nenhum par válido no manifesto")

    scale = calibrate_robust_scale(
        np.concatenate([case["real"][case["roi"]] for case in accepted]),
        tukey_k=args.tukey_k,
        levels=args.levels,
    )
    rows = []
    for case in accepted:
        for image_type in ("real", "synthetic"):
            rows.append({"case_id": case["case_id"], "image_type": image_type,
                         **extract_glcm(case[image_type], case["roi"], scale)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "glcm_features.csv", rows)
    _write_csv(args.output_dir / "rejected_cases.csv", rejected)
    with (args.output_dir / "glcm_calibration.json").open("w", encoding="utf-8") as file:
        json.dump({"accepted_pairs": len(accepted), "rejected_pairs": len(rejected),
                   "erosion_radius": args.erosion_radius, "mask_label": args.mask_label,
                   **scale.as_dict()}, file, indent=2)


if __name__ == "__main__":
    main()
