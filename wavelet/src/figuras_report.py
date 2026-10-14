"""Figuras do report: o brilho sobe e as features wavelet não percebem.

1. `exemplo_brilho_sem_wavelet.png`: um corte gerado em que o pulmão clareou,
   lado a lado com a original, com a densidade e as features wavelet (energia e
   entropia, como o framework calcula) em relação à original.
2. `wavelet_resposta_so_brilho.png`: a coluna "Brilho" de
   `wavelet_brightness.py` na horizontal (lê `varredura_por_caso.csv`).
"""
from __future__ import annotations

import argparse
import csv
from types import SimpleNamespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .density_features import density_features, density_limits, density_status, reference_interval
from .wavelet_diagnostico import load_cases, raw_readings
from .wavelet_features_fullimage import wavelet_energy_entropy_fullimage

DENSITY_TEXT = {"ok": "normal", "brighter": "clareou", "too_bright": "claro demais", "darker": "escureceu", "too_dark": "escuro demais"}


def save_example(case, interval, limits, output: Path) -> None:
    original, generated = (density_features(image, case.lung_mask, interval) for image in (case.original, case.synthetic))
    status = density_status(generated, original, limits)
    wavelet = [wavelet_energy_entropy_fullimage(image, case.lung_mask, interval["p1"], interval["p99"], "haar", 2, 0.9) for image in (case.original, case.synthetic)]
    energy, entropy = (100 * wavelet[1][key] / wavelet[0][key] for key in ("energy", "entropy"))
    # média da LL na escala da faixa P1–P99 (0 = P1, 1 = P99), mesma leitura de wavelet_diagnostico
    settings = SimpleNamespace(wavelet="haar", min_tissue=0.9)
    ll_original, ll_generated = (raw_readings(image, case.lung_mask, interval, settings)["ll"] for image in (case.original, case.synthetic))

    rows, cols = np.where(case.lung_mask)
    crop = (slice(rows.min() - 6, rows.max() + 6), slice(cols.min() - 6, cols.max() + 6))
    fig = plt.figure(figsize=(12, 5))
    grid = fig.add_gridspec(1, 3, width_ratios=(1, 1, 0.9))
    for column, (title, image) in enumerate((("Original", case.original), ("Gerada", case.synthetic))):
        axis = fig.add_subplot(grid[0, column])
        axis.imshow(image[crop], cmap="gray", vmin=interval["p1"], vmax=interval["p99"])
        axis.contour(case.lung_mask[crop], levels=[0.5], colors="#00a36c", linewidths=0.7)
        axis.set_title(title, fontsize=13)
        axis.axis("off")

    axis = fig.add_subplot(grid[0, 2])
    axis.axis("off")
    lines = (
        ("Densidade", DENSITY_TEXT[status], "#c0392b"),
        ("LL da wavelet (brilho, 0 = P1, 1 = P99)", f"{ll_original:.2f} → {ll_generated:.2f}".replace(".", ","), "#c0392b"),
        ("Energia wavelet", f"{energy:.0f}% da original", "#333333"),
        ("Entropia wavelet", f"{entropy:.0f}% da original", "#333333"),
    )
    for row, (label, value, color) in enumerate(lines):
        y = 0.86 - row * 0.24
        axis.text(0.02, y, label, fontsize=12, color="#555555", transform=axis.transAxes)
        axis.text(0.02, y - 0.1, value, fontsize=17, fontweight="bold", color=color, transform=axis.transAxes)
    fig.suptitle("O pulmão clareou: a LL acompanha, mas a energia e a entropia quase não mudam", fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_brightness_row(sweep_csv: Path, output: Path) -> None:
    rows = [row for row in csv.DictReader(sweep_csv.open()) if row["family"] == "brilho"]
    xs = sorted({float(row["x"]) for row in rows})
    panels = (
        ("pipeline|energy_pct", "sem corte|energy_pct", "Energia de detalhe (% da original)", (40, 115)),
        ("pipeline|entropy_pct", "sem corte|entropy_pct", "Entropia de detalhe (% da original)", (40, 115)),
        ("pipeline|ll", "sem corte|ll", "Média da LL (0 = P1, 1 = P99)", None),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for axis, (pipeline, pure, title, limits) in zip(axes, panels):
        for key, label, color in ((pure, "wavelet pura", "#F58518"), (pipeline, "com o corte P1–P99 do pipeline", "#4C78A8")):
            values = [[float(row[key]) for row in rows if float(row["x"]) == x] for x in xs]
            median = [np.median(v) for v in values]
            low, high = zip(*(np.percentile(v, (25, 75)) for v in values))
            axis.plot(xs, median, "o-", color=color, label=label, markersize=3)
            axis.fill_between(xs, low, high, color=color, alpha=.2)
        if limits:
            axis.axhline(100, color="#888888", linestyle=":", linewidth=1)
            axis.set_ylim(*limits)
        axis.axvline(0, color="#bbbbbb", linewidth=0.8)
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("brilho somado (fração da faixa P1–P99)")
    axes[0].legend(fontsize=8, loc="lower center")
    fig.suptitle("Brilho: a wavelet pura não muda energia nem entropia; só a LL acompanha (mediana e IQR de 37 cortes)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="dataset padrão (Dsr), usado para a referência")
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--examples-dir", type=Path, required=True, help="dataset de onde sai o exemplo (Dgen)")
    parser.add_argument("--case-id", type=str, required=True, help="corte do exemplo")
    parser.add_argument("--sweep-csv", type=Path, required=True, help="varredura_por_caso.csv de wavelet_brightness.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_cases(args.data_dir, args.mask_dir, 2)
    interval = reference_interval([(case.original, case.lung_mask) for case in cases])
    limits = density_limits([density_features(case.original, case.lung_mask, interval) for case in cases])
    example = next(case for case in load_cases(args.examples_dir, args.mask_dir, 2) if case.paths.case_id == args.case_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_example(example, interval, limits, args.output_dir / "exemplo_brilho_sem_wavelet.png")
    save_brightness_row(args.sweep_csv, args.output_dir / "wavelet_resposta_so_brilho.png")


if __name__ == "__main__":
    main()
