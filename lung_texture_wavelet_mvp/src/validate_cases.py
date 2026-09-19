"""Regras transparentes para aceitar cortes pulmonares no MVP."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import disk, erosion

from .load_data import CasePaths, load_image


@dataclass
class ValidCase:
    paths: CasePaths
    original: np.ndarray
    synthetic: np.ndarray
    lung_mask: np.ndarray
    area_fraction: float
    component_count: int


@dataclass
class RejectedCase:
    paths: CasePaths
    reason: str


def validate_case(
    paths: CasePaths,
    mask_label: float = 1.0,
    erosion_radius: int = 2,
    min_component_pixels: int = 800,
    min_lung_area: float = 0.03,
    max_lung_area: float = 0.35,
) -> ValidCase | RejectedCase:
    """Valida correspondência, geometria da máscara e presença dos dois pulmões.

    A regra não substitui revisão clínica: ela é um filtro operacional para
    manter apenas cortes torácicos centrais com região pulmonar utilizável.
    """
    if paths.synthetic is None or not paths.synthetic.exists():
        return RejectedCase(paths, "imagem sintética ausente")
    if paths.mask is None or not paths.mask.exists():
        return RejectedCase(paths, "máscara ausente")
    try:
        original, synthetic, raw_mask = load_image(paths.original), load_image(paths.synthetic), load_image(paths.mask)
    except Exception as error:
        return RejectedCase(paths, f"falha ao carregar: {error}")
    if original.shape != synthetic.shape or original.shape != raw_mask.shape:
        return RejectedCase(paths, f"dimensões não correspondem: original={original.shape}, sintética={synthetic.shape}, máscara={raw_mask.shape}")
    if not (np.isfinite(original).all() and np.isfinite(synthetic).all() and np.isfinite(raw_mask).all()):
        return RejectedCase(paths, "há valores não finitos")

    lung = np.isclose(raw_mask, mask_label)
    components = [region for region in regionprops(label(lung)) if region.area >= min_component_pixels]
    area_fraction = float(lung.mean())
    if len(components) < 2:
        return RejectedCase(paths, "a máscara não tem dois componentes pulmonares relevantes")
    if not min_lung_area <= area_fraction <= max_lung_area:
        return RejectedCase(paths, f"área pulmonar fora da faixa operacional ({area_fraction:.1%})")
    eroded = erosion(lung, footprint=disk(erosion_radius)) if erosion_radius else lung
    if int(eroded.sum()) < min_component_pixels:
        return RejectedCase(paths, "a erosão deixou poucos pixels pulmonares")
    return ValidCase(paths, original, synthetic, eroded, area_fraction, len(components))
