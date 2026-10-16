"""Régua única: a textura (GLCM e Wavelet) fica cega ao brilho?

Compara duas réguas de clipping/normalização, aplicadas igualmente às duas métricas:

- `P1–P99`: a régua original da Wavelet (percentis dos pixels pulmonares reais);
- `Tukey k=3`: a régua do protocolo GLCM (cercas externas de Tukey nos pixels reais).

Parte 1 (simulado): soma brilho puro às originais e mede quanto cada métrica de
textura muda, em unidades de IQR entre originais (|efeito| > 1 = mais que a
diferença entre dois pulmões reais). Brilho puro não muda a textura, então o
ideal é efeito ~0. Contraste ×0,5 e suavização entram como controle: aí a
textura *deve* reagir.

Parte 2 (dados reais): correlação de Spearman entre cada métrica (sintética ÷
original) e o Δμ do par, mais a saturação no piso e no teto de cada régua.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

from wavelet.src.density_features import reference_interval
from wavelet.src.load_data import CasePaths, discover_cases
from wavelet.src.run_pipeline import write_csv
from wavelet.src.validate_cases import ValidCase, validate_case
from wavelet.src.wavelet_diagnostico import raw_readings
from frameworks.texture_evaluator.glcm.src import RobustScale, calibrate_robust_scale, extract_glcm

METRICS = ("glcm_contrast", "glcm_energy", "wavelet_fine", "wavelet_coarse", "wavelet_ratio", "wavelet_entropy")
SETTINGS = SimpleNamespace(wavelet="haar", min_tissue=.9)


def rulers(valid) -> dict[str, RobustScale]:
    pixels = np.concatenate([c.original[c.lung_mask] for c in valid])
    tukey = calibrate_robust_scale(pixels, tukey_k=3, levels=32)
    interval = reference_interval([(c.original, c.lung_mask) for c in valid])
    percentile = RobustScale(tukey.q1, tukey.q3, tukey.iqr, interval["p1"], interval["p99"], tukey_k=float("nan"), levels=32)
    return {"P1–P99": percentile, "Tukey k=3": tukey}


def measure(image, mask, scale: RobustScale) -> dict[str, float]:
    glcm = extract_glcm(image, mask, scale)
    wavelet = raw_readings(image, mask, {"p1": scale.clip_low, "p99": scale.clip_high}, SETTINGS)
    return {
        "glcm_contrast": glcm["glcm_contrast"], "glcm_energy": glcm["glcm_energy"],
        "wavelet_fine": wavelet["fine"], "wavelet_coarse": wavelet["coarse"],
        "wavelet_ratio": wavelet["fine"] / wavelet["coarse"], "wavelet_entropy": wavelet["entropy"],
        "floor_pct": glcm["saturation_tone_0_pct"], "ceiling_pct": glcm["saturation_tone_31_pct"],
    }


def perturbations(iqr: float):
    def offset(k):
        return lambda image, mask: image + k * iqr

    def gain(a):
        return lambda image, mask: np.median(image[mask]) + a * (image - np.median(image[mask]))

    return [
        ("brilho −1 IQR", offset(-1)), ("brilho −0,5 IQR", offset(-.5)),
        ("brilho +0,5 IQR", offset(.5)), ("brilho +1 IQR", offset(1)),
        ("controle: contraste ×0,5", gain(.5)), ("controle: suavização σ=1", lambda image, mask: gaussian_filter(image, 1)),
    ]


def save_heatmap(effects: dict[str, dict[str, dict[str, float]]], output: Path) -> None:
    names = list(effects)
    labels = list(effects[names[0]])
    fig, axes = plt.subplots(1, len(names), figsize=(6.2 * len(names), 0.5 * len(labels) + 2), sharey=True)
    for axis, name in zip(axes, names):
        matrix = np.array([[effects[name][label][metric] for metric in METRICS] for label in labels])
        axis.imshow(np.clip(matrix, -5, 5), cmap="RdBu_r", vmin=-5, vmax=5, aspect="auto")
        for (row, column), value in np.ndenumerate(matrix):
            axis.text(column, row, f"{value:+.1f}", ha="center", va="center", fontsize=8, color="white" if abs(value) > 3 else "black")
        axis.set_xticks(range(len(METRICS)), [metric.replace("_", "\n", 1) for metric in METRICS], fontsize=8)
        axis.set_yticks(range(len(labels)), labels, fontsize=9)
        axis.axhline(3.5, color="#333333", linewidth=1)
        axis.set_title(f"Régua {name}", fontsize=11)
    fig.suptitle("Quanto cada defeito mexe na textura (mudança mediana / IQR entre originais; brilho puro deveria dar ~0)", fontsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--mask-dir", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    valid = []
    for paths in discover_cases(a.data_dir):
        if a.mask_dir:
            paths = CasePaths(paths.case_id, paths.original, paths.synthetic, a.mask_dir / f"{paths.case_id}_mask.tiff")
        result = validate_case(paths, erosion_radius=2)
        if isinstance(result, ValidCase):
            valid.append(result)
    scales = rulers(valid)
    items = perturbations(scales["Tukey k=3"].iqr)

    rows, effects, real = [], {}, {}
    for name, scale in scales.items():
        base = {c.paths.case_id: measure(c.original, c.lung_mask, scale) for c in valid}
        iqr = {m: float(np.subtract(*np.percentile([b[m] for b in base.values()], (75, 25)))) for m in METRICS}
        effects[name] = {}
        for label, apply in items:
            changes = {m: [] for m in METRICS}
            for c in valid:
                values = measure(apply(c.original, c.lung_mask), c.lung_mask, scale)
                for m in METRICS:
                    changes[m].append((values[m] - base[c.paths.case_id][m]) / max(iqr[m], 1e-12))
                rows.append({"ruler": name, "perturbation": label, "case_id": c.paths.case_id, **values})
            effects[name][label] = {m: float(np.median(v)) for m, v in changes.items()}
        # dados reais: a razão sintética ÷ original acompanha o Δμ do par?
        synthetic = {c.paths.case_id: measure(c.synthetic, c.lung_mask, scale) for c in valid}
        delta = [float(c.synthetic[c.lung_mask].mean() - c.original[c.lung_mask].mean()) for c in valid]
        ratios = {m: [synthetic[c.paths.case_id][m] / base[c.paths.case_id][m] for c in valid] for m in METRICS}
        real[name] = {
            "clip": [scale.clip_low, scale.clip_high],
            "spearman_vs_delta_mu": {m: float(spearmanr(ratios[m], delta)[0]) for m in METRICS},
            "median_ratio": {m: float(np.median(v)) for m, v in ratios.items()},
            "floor_pct_median": {kind: float(np.median([v["floor_pct"] for v in group.values()])) for kind, group in (("original", base), ("synthetic", synthetic))},
            "ceiling_pct_median": {kind: float(np.median([v["ceiling_pct"] for v in group.values()])) for kind, group in (("original", base), ("synthetic", synthetic))},
            "floor_pct_max_synthetic": float(max(v["floor_pct"] for v in synthetic.values())),
        }

    figures = a.output_dir / "figuras"
    figures.mkdir(parents=True, exist_ok=True)
    write_csv(a.output_dir / "medidas_por_caso.csv", rows)
    (a.output_dir / "resumo.json").write_text(json.dumps({"n_pairs": len(valid), "effect_iqr": effects, "real_data": real}, indent=2, ensure_ascii=False), encoding="utf-8")
    save_heatmap(effects, figures / "efeito_brilho_por_regua.png")
    print(json.dumps({"effect_iqr": effects, "real_data": real}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
