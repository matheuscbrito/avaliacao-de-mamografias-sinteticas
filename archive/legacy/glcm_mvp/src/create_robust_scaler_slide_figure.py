"""Cria uma figura didática do Robust Scaler a partir da coorte real."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from src.load_data import discover_cases
from src.validate_cases import ValidCase, validate_case


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dados" / "Dgen"
OUTPUT = ROOT / "slides" / "imagens" / "robust_scaler_como_funciona.png"


def pooled_real_roi_pixels() -> np.ndarray:
    pixels: list[np.ndarray] = []
    for paths in discover_cases(DATA):
        result = validate_case(paths)
        if isinstance(result, ValidCase):
            pixels.append(result.original[result.lung_mask])
    return np.concatenate(pixels)


def main() -> None:
    values = pooled_real_roi_pixels()
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    low, high = q1 - 3 * iqr, q3 + 3 * iqr

    navy, teal, orange, gray = "#102A43", "#007C83", "#E76F00", "#7B8794"
    fig = plt.figure(figsize=(12.5, 6.7), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.08], wspace=0.28)
    axes = [fig.add_subplot(grid[0, index]) for index in range(2)]
    bins = np.linspace(-1.65, 1.75, 85)

    # 1. Estatísticas robustas da distribuição real
    ax = axes[0]
    ax.hist(values, bins=bins, density=True, color="#B7C9DB", edgecolor="white", linewidth=.25)
    ax.axvspan(q1, q3, color=teal, alpha=.18, label="50% central (IQR)")
    ax.axvline(q1, color=teal, linewidth=2)
    ax.axvline(median, color=navy, linewidth=1.6, linestyle=":")
    ax.axvline(q3, color=teal, linewidth=2)
    ax.set_title("1. Distribuição dos pixels reais", loc="left", fontsize=15, fontweight="bold", color=navy)
    ax.text(.02, .93, "ROI pulmonar das 37 CTs reais", transform=ax.transAxes,
            va="top", fontsize=10.5, color=navy)
    ax.text((q1 + q3) / 2, ax.get_ylim()[1] * .57, "Faixa interquartil\nentre Q1 e Q3",
            ha="center", va="bottom", fontsize=10, color=navy,
            bbox=dict(boxstyle="round,pad=.28", facecolor="white", edgecolor="none", alpha=.94))
    ax.text((q1 + q3) / 2, ax.get_ylim()[1] * .11, "IQR = Q3 − Q1", ha="center", color=teal,
            fontsize=11, fontweight="bold")
    ax.set_xlabel("Intensidade na ROI pulmonar")
    ax.set_ylabel("Densidade de pixels")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_yticks([])

    # 2. Cercas de Tukey baseadas nesses valores
    ax = axes[1]
    ax.hist(values, bins=bins, density=True, color="#D9E2EC", edgecolor="white", linewidth=.25)
    ax.axvspan(low, high, color=teal, alpha=.12)
    ax.axvline(low, color=orange, linewidth=2.4)
    ax.axvline(high, color=orange, linewidth=2.4)
    ax.axvline(q1, color=teal, linewidth=1.3, linestyle="--")
    ax.axvline(q3, color=teal, linewidth=1.3, linestyle="--")
    ax.set_title("2. Definir os limites de Tukey", loc="left", fontsize=15, fontweight="bold", color=navy)
    ax.text(.02, .93, "Piso = Q1 − 3 × IQR    |    Teto = Q3 + 3 × IQR", transform=ax.transAxes,
            va="top", fontsize=10.5, color=navy)
    ax.annotate(f"Piso\n{low:.2f}", xy=(low, 0), xytext=(low, ax.get_ylim()[1] * .48), ha="center",
                color=orange, fontweight="bold", arrowprops=dict(arrowstyle="-", color=orange, lw=1.3))
    ax.annotate(f"Teto\n{high:.2f}", xy=(high, 0), xytext=(high, ax.get_ylim()[1] * .48), ha="center",
                color=orange, fontweight="bold", arrowprops=dict(arrowstyle="-", color=orange, lw=1.3))
    ax.set_xlabel("Intensidade na ROI pulmonar")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_yticks([])

    fig.suptitle("Robust Scaler: limites definidos pela distribuição real dos pixels", x=.06, y=.99, ha="left",
                 fontsize=19, fontweight="bold", color=navy)
    fig.text(.06, .015, "Dados reais: ROI pulmonar erodida | Calibração por coorte | Tukey k = 3,0",
             fontsize=10.5, color=gray)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
