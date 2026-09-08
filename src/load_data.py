"""Descoberta e carregamento dos pares original, sintética e máscara."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CasePaths:
    case_id: str
    original: Path
    synthetic: Path | None
    mask: Path | None


def load_image(path: Path) -> np.ndarray:
    """Lê uma imagem TIFF/PNG como matriz 2D float."""
    image = np.asarray(Image.open(path), dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"{path.name}: esperava imagem 2D, recebeu {image.shape}")
    return image


def _flat_cases(data_dir: Path) -> list[CasePaths]:
    originals = sorted(
        path for path in data_dir.glob("*.tiff")
        if not path.name.endswith(("_generate.tiff", "_mask.tiff"))
    )
    return [
        CasePaths(
            case_id=original.stem,
            original=original,
            synthetic=original.with_name(f"{original.stem}_generate.tiff"),
            mask=original.with_name(f"{original.stem}_mask.tiff"),
        )
        for original in originals
    ]


def _structured_cases(data_dir: Path) -> list[CasePaths]:
    original_dir, synthetic_dir, mask_dir = (data_dir / name for name in ("original", "synthetic", "masks"))
    originals = sorted(path for path in original_dir.iterdir() if path.suffix.lower() in {".tif", ".tiff", ".png"})
    cases: list[CasePaths] = []
    for original in originals:
        candidates_synthetic = [synthetic_dir / original.name, synthetic_dir / f"{original.stem}_generate{original.suffix}"]
        candidates_mask = [mask_dir / original.name, mask_dir / f"{original.stem}_mask{original.suffix}"]
        cases.append(CasePaths(
            case_id=original.stem,
            original=original,
            synthetic=next((path for path in candidates_synthetic if path.exists()), candidates_synthetic[0]),
            mask=next((path for path in candidates_mask if path.exists()), candidates_mask[0]),
        ))
    return cases


def discover_cases(data_dir: Path) -> list[CasePaths]:
    """Aceita o dataset atual (TIFFs em uma pasta) ou a estrutura original/synthetic/masks."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {data_dir}")
    if all((data_dir / name).is_dir() for name in ("original", "synthetic", "masks")):
        return _structured_cases(data_dir)
    return _flat_cases(data_dir)
