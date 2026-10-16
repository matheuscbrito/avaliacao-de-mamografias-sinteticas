"""A camada de decisão acerta defeitos conhecidos?

Cada original vira a sua própria "sintética" com um defeito conhecido, e o par
passa pelo MVP inteiro (`evaluate_pair` + `decide`). Para cada defeito, conta
quantos cortes saem `aprovada`, `revisar` ou `rejeitar` e qual leitura apontou.

O brilho é somado em múltiplos do IQR das médias pulmonares das originais, a
mesma unidade do filtro do contrato (revisão acima de 1, rejeição acima de 2).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from wavelet.src.load_data import CasePaths, discover_cases
from wavelet.src.run_pipeline import write_csv
from wavelet.src.validate_cases import ValidCase, validate_case
from frameworks.texture_evaluator.decision import FINAL_STATUSES
from frameworks.texture_evaluator.report import FINAL
from frameworks.texture_evaluator.run_mvp import calibrate, evaluate_pair


def defects(ctx):
    width = ctx.scale.clip_high - ctx.scale.clip_low
    rng = np.random.default_rng(0)

    def offset(k):
        return lambda image, mask: image + k * ctx.mean_iqr

    def gain(a):
        return lambda image, mask: np.median(image[mask]) + a * (image - np.median(image[mask]))

    def washout(w):
        def apply(image, mask):
            out = image.copy()
            out[mask] = (1 - w) * image[mask] + w * ctx.interval["p99"]
            return out
        return apply

    return [
        ("sem defeito", lambda image, mask: image.copy()),
        ("brilho +0,5", offset(.5)), ("brilho +1,5", offset(1.5)), ("brilho +2,5", offset(2.5)),
        ("brilho −1,5", offset(-1.5)),
        ("contraste ×0,5", gain(.5)), ("contraste ×1,5", gain(1.5)),
        ("suavização σ=1", lambda image, mask: gaussian_filter(image, 1)),
        ("suavização σ=2", lambda image, mask: gaussian_filter(image, 2)),
        ("ruído 2%", lambda image, mask: image + rng.normal(0, .02 * width, image.shape)),
        ("ruído 5%", lambda image, mask: image + rng.normal(0, .05 * width, image.shape)),
        ("lavagem 50%", washout(.5)),
    ]


def save_figure(summary: dict, output: Path) -> None:
    labels = list(summary)
    fig, axis = plt.subplots(figsize=(11, 0.45 * len(labels) + 1.5))
    left = np.zeros(len(labels))
    for status in FINAL_STATUSES:
        values = np.array([summary[label]["final"].get(status, 0) for label in labels])
        axis.barh(labels, values, left=left, color=FINAL[status], label=status)
        for row, (start, value) in enumerate(zip(left, values)):
            if value:
                axis.text(start + value / 2, row, str(value), ha="center", va="center", fontsize=8, color="white")
        left += values
    for row, label in enumerate(labels):
        axis.text(left[row] + .5, row, summary[label]["texture"], va="center", fontsize=8, color="#444444")
    axis.invert_yaxis()
    axis.set_xlim(0, left.max() * 1.55)
    axis.set_xlabel("Número de cortes (texto: veredito Wavelet mais comum)")
    axis.legend(loc="lower right", fontsize=8)
    axis.set_title("Veredito final para defeitos conhecidos aplicados às originais", fontsize=11)
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
    ctx = calibrate(valid)

    rows, summary = [], {}
    for label, apply in defects(ctx):
        group = [evaluate_pair(c.paths.case_id, c.original, apply(c.original, c.lung_mask), c.lung_mask, ctx) for c in valid]
        rows += [{"defect": label, **row} for row in group]
        texture = Counter(row["wavelet_verdict"] for row in group).most_common(1)[0]
        summary[label] = {
            "final": dict(Counter(row["final_status"] for row in group)),
            "brightness_filter": dict(Counter(row["brightness_status"] for row in group)),
            "density": dict(Counter(row["density_status"] for row in group)),
            "texture": f"{texture[0]} ({texture[1]}/{len(group)})",
            "disagreement": sum(row["texture_agreement"] == "discordam" for row in group),
        }

    figures = a.output_dir / "figuras"
    figures.mkdir(parents=True, exist_ok=True)
    write_csv(a.output_dir / "decisao_por_caso.csv", rows)
    (a.output_dir / "resumo.json").write_text(json.dumps({"n_pairs": len(valid), "by_defect": summary}, indent=2, ensure_ascii=False), encoding="utf-8")
    save_figure(summary, figures / "decisao_defeitos_conhecidos.png")
    for label, values in summary.items():
        print(f"{label:16} {values['final']}  filtro={values['brightness_filter']}  dens={values['density']}  wav={values['texture']}  disc={values['disagreement']}")


if __name__ == "__main__":
    main()
