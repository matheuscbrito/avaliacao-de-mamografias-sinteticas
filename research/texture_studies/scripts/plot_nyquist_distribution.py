"""Figuras descritivas do escore experimental próximo a Nyquist.

Este script apenas visualiza os valores já armazenados em qc.parquet. Ele não
valida a interpretação física nem a implementação do estimador.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[1]
QC_PATH = ROOT / "outputs" / "gen-fid" / "qc.parquet"
MANIFEST_PATH = ROOT / "outputs" / "gen-fid" / "manifest.csv"
PLOTS_DIR = ROOT / "plots"

REAL = "#2878B5"
SYN = "#E67E22"
DELTA = "#665191"


def load_values() -> pd.DataFrame:
    qc = pd.read_parquet(QC_PATH)
    wide = (
        qc.loc[qc["flag"].isin(["nyquist_excess_real", "nyquist_excess_syn"])]
        .pivot(index="pair_id", columns="flag", values="value")
        .reset_index()
    )
    manifest = pd.read_csv(MANIFEST_PATH)[["pair_id", "batch", "patient_id"]]
    out = wide.merge(manifest, on="pair_id", validate="one_to_one")
    out["delta"] = out["nyquist_excess_syn"] - out["nyquist_excess_real"]
    return out


def kde(ax: plt.Axes, values: np.ndarray, color: str, label: str) -> None:
    grid = np.linspace(-1.0, 3.4, 600)
    density = gaussian_kde(values)(grid)
    ax.fill_between(grid, density, color=color, alpha=0.27)
    ax.plot(grid, density, color=color, linewidth=2.4, label=label)


def distribution_figure(df: pd.DataFrame) -> None:
    real = df["nyquist_excess_real"].to_numpy()
    syn = df["nyquist_excess_syn"].to_numpy()
    delta = df["delta"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)

    ax = axes[0]
    kde(ax, real, REAL, f"Real — mediana {np.median(real):.2f}")
    kde(ax, syn, SYN, f"Sintético — mediana {np.median(syn):.2f}")
    rng = np.random.default_rng(42)
    ax.scatter(real, rng.uniform(-0.045, -0.015, len(real)), s=9, color=REAL, alpha=0.42, edgecolors="none")
    ax.scatter(syn, rng.uniform(-0.095, -0.065, len(syn)), s=9, color=SYN, alpha=0.42, edgecolors="none")
    ax.axvline(np.median(real), color=REAL, linestyle="--", linewidth=1.3)
    ax.axvline(np.median(syn), color=SYN, linestyle="--", linewidth=1.3)
    ax.set_title("A. Distribuição observada nas 225 ROIs", fontweight="bold")
    ax.set_xlabel("Escore experimental: log10 da razão pico/fundo")
    ax.set_ylabel("Densidade estimada")
    ax.set_ylim(bottom=-0.12)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    bins = np.linspace(0.25, 3.55, 25)
    ax.hist(delta, bins=bins, color=DELTA, alpha=0.72, edgecolor="white", linewidth=0.7)
    ax.axvline(np.median(delta), color="#35245C", linestyle="--", linewidth=2)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_title("B. Diferença pareada: sintético − real", fontweight="bold")
    ax.set_xlabel("Diferença do escore em log10")
    ax.set_ylabel("Número de pares")
    ax.text(
        0.97,
        0.94,
        f"225/225 diferenças positivas\nmediana = {np.median(delta):.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.55)

    fig.suptitle(
        "Separação real–sintético observada pelo estimador atual",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.025,
        "Resultado descritivo e pós-hoc: a interpretação física e a fórmula do estimador ainda serão validadas.",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    for ext in ("png", "svg"):
        fig.savefig(PLOTS_DIR / f"nyquist_excess_distribuicao.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def distribution_only_figure(df: pd.DataFrame) -> None:
    real = df["nyquist_excess_real"].to_numpy()
    syn = df["nyquist_excess_syn"].to_numpy()
    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    kde(ax, real, REAL, f"Real — mediana {np.median(real):.2f}")
    kde(ax, syn, SYN, f"Sintético — mediana {np.median(syn):.2f}")
    rng = np.random.default_rng(42)
    ax.scatter(real, rng.uniform(-0.045, -0.015, len(real)), s=10, color=REAL, alpha=0.42, edgecolors="none")
    ax.scatter(syn, rng.uniform(-0.095, -0.065, len(syn)), s=10, color=SYN, alpha=0.42, edgecolors="none")
    ax.axvline(np.median(real), color=REAL, linestyle="--", linewidth=1.4)
    ax.axvline(np.median(syn), color=SYN, linestyle="--", linewidth=1.4)
    ax.set_title("Distribuição observada nas 225 ROIs", fontsize=15, fontweight="bold")
    ax.set_xlabel("Escore experimental: log10 da razão pico/fundo")
    ax.set_ylabel("Densidade estimada")
    ax.set_ylim(bottom=-0.12)
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.55)
    fig.text(
        0.5,
        -0.025,
        "Resultado descritivo e pós-hoc; o estimador ainda será validado.",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    for ext in ("png", "svg"):
        fig.savefig(PLOTS_DIR / f"nyquist_excess_distribuicoes.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def paired_delta_figure(df: pd.DataFrame) -> None:
    delta = df["delta"].to_numpy()
    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    bins = np.linspace(0.25, 3.55, 25)
    ax.hist(delta, bins=bins, color=DELTA, alpha=0.76, edgecolor="white", linewidth=0.8)
    ax.axvline(np.median(delta), color="#35245C", linestyle="--", linewidth=2.2)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_title("Diferença pareada: sintético − real", fontsize=15, fontweight="bold")
    ax.set_xlabel("Diferença do escore em log10")
    ax.set_ylabel("Número de pares")
    ax.text(
        0.97,
        0.94,
        f"225/225 diferenças positivas\nmediana = {np.median(delta):.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.55)
    fig.text(
        0.5,
        -0.025,
        "Cada valor corresponde à diferença dentro de um par real–sintético.",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    for ext in ("png", "svg"):
        fig.savefig(PLOTS_DIR / f"nyquist_excess_diferenca_pareada.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def batch_figure(df: pd.DataFrame) -> None:
    batches = sorted(df["batch"].unique())
    fig, ax = plt.subplots(figsize=(10.8, 5.6), constrained_layout=True)
    positions = []
    data = []
    colors = []
    labels = []
    for idx, batch in enumerate(batches):
        subset = df[df["batch"] == batch]
        positions.extend([idx * 3 + 1, idx * 3 + 2])
        data.extend([subset["nyquist_excess_real"], subset["nyquist_excess_syn"]])
        colors.extend([REAL, SYN])
        labels.append(f"Lote {batch}\n(n={len(subset)})")

    parts = ax.violinplot(data, positions=positions, widths=0.8, showextrema=False)
    for body, color in zip(parts["bodies"], colors, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.25)

    rng = np.random.default_rng(7)
    for values, pos, color in zip(data, positions, colors, strict=True):
        vals = np.asarray(values)
        ax.scatter(rng.normal(pos, 0.075, len(vals)), vals, s=12, color=color, alpha=0.38, edgecolors="none")
        q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
        ax.vlines(pos, q1, q3, color=color, linewidth=5)
        ax.scatter([pos], [med], color="white", edgecolor=color, linewidth=1.5, s=42, zorder=4)

    centers = [idx * 3 + 1.5 for idx in range(len(batches))]
    ax.set_xticks(centers, labels)
    ax.set_ylabel("Escore experimental: log10 da razão pico/fundo")
    ax.set_title("Distribuição por lote de geração", fontsize=14, fontweight="bold")
    ax.scatter([], [], color=REAL, label="Real")
    ax.scatter([], [], color=SYN, label="Sintético")
    ax.legend(frameon=False, ncols=2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.55)
    fig.text(
        0.5,
        -0.02,
        "A separação aparece nos três lotes, mas o escore real também varia entre lotes; isso exige controle estatístico.",
        ha="center",
        fontsize=9.5,
        color="#555555",
    )
    for ext in ("png", "svg"):
        fig.savefig(PLOTS_DIR / f"nyquist_excess_por_lote.{ext}", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    values = load_values()
    distribution_figure(values)
    distribution_only_figure(values)
    paired_delta_figure(values)
    batch_figure(values)
    print(f"Figuras gravadas em {PLOTS_DIR}")
