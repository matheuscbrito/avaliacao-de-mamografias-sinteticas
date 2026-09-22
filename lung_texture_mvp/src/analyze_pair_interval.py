"""Teste de sensibilidade: discretização robusta conjunta por par.

Para cada par, os percentis P1 e P99 são calculados sobre a união dos pixels
da ROI pulmonar da original e da sintética. O mesmo intervalo é então aplicado
às duas imagens antes do GLCM, preservando a comparabilidade dentro do par.

Execute a partir de ``lung_texture_mvp``:
    ../.venv/bin/python -m src.analyze_pair_interval
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .glcm_features import glcm_contrast_energy
from .make_audit_report import case_arrays


INPUT = Path("outputs/comparison_results.csv")
OUTPUT = Path("outputs/intervalo_por_par")
PERCENTILES = (1.0, 99.0)
LEVELS = 32

# Limites fixos, iguais aos usados visualmente no gráfico do protocolo global.
AXIS_LIMITS = {
    "contrast": (1.75, 11.00),
    "energy": (-0.002, 0.190),
}


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def calculate(rows: list[dict[str, str]]) -> list[dict[str, float | str]]:
    result: list[dict[str, float | str]] = []
    for row in rows:
        original, synthetic, mask = case_arrays(row["case_id"])
        pair_pixels = np.concatenate((original[mask], synthetic[mask]))
        clip_low, clip_high = (float(value) for value in np.percentile(pair_pixels, PERCENTILES))
        original_metrics = glcm_contrast_energy(original, mask, clip_low, clip_high, LEVELS)
        synthetic_metrics = glcm_contrast_energy(synthetic, mask, clip_low, clip_high, LEVELS)
        result.append({
            "case_id": row["case_id"],
            "clip_low_pair_p1": clip_low,
            "clip_high_pair_p99": clip_high,
            "original_contrast": original_metrics["contrast"],
            "synthetic_contrast": synthetic_metrics["contrast"],
            "original_energy": original_metrics["energy"],
            "synthetic_energy": synthetic_metrics["energy"],
        })
    return result


def draw_pair_scatter(axes, rows: list[dict[str, float | str]], row_title: str) -> None:
    for axis, feature, title in zip(axes, ("contrast", "energy"), ("Contraste", "Energia")):
        original = np.array([float(row[f"original_{feature}"]) for row in rows])
        synthetic = np.array([float(row[f"synthetic_{feature}"]) for row in rows])
        low, high = AXIS_LIMITS[feature]
        axis.scatter(original, synthetic, color="#4C78A8", alpha=0.8)
        axis.plot((low, high), (low, high), "--", color="#888888", label="sintética = original")
        axis.set(title=title, xlabel="Original", ylabel="Sintética", xlim=(low, high), ylim=(low, high))
        axis.legend(fontsize=8)


def save_plot(rows: list[dict[str, float | str]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    draw_pair_scatter(axes, rows, "Intervalo por par")
    fig.suptitle("GLCM com intervalo P1–P99 conjunto por par", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT / "original_vs_sintetica_intervalo_por_par.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_comparison_plot(baseline: list[dict[str, str]], pair_rows: list[dict[str, float | str]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.6))
    draw_pair_scatter(axes[0], baseline, "Global")
    draw_pair_scatter(axes[1], pair_rows, "Por par")
    axes[0, 0].set_ylabel("Sintética\n\nProtocolo global", labelpad=12)
    axes[1, 0].set_ylabel("Sintética\n\nIntervalo por par", labelpad=12)
    fig.suptitle("Comparação em escala idêntica: global versus intervalo P1–P99 por par", y=0.98, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUTPUT / "comparacao_global_vs_intervalo_por_par.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    baseline = list(csv.DictReader(INPUT.open(encoding="utf-8")))
    pair_rows = calculate(baseline)
    write_csv(OUTPUT / "features_glcm_intervalo_por_par.csv", pair_rows)
    save_plot(pair_rows)
    save_comparison_plot(baseline, pair_rows)

    summary = {
        "method": "P1–P99 calculado na união dos pixels da ROI da original e da sintética, por par",
        "levels": LEVELS,
        "n_pairs": len(pair_rows),
        "axis_limits": AXIS_LIMITS,
        "medians": {
            feature: {
                "original": float(np.median([float(row[f"original_{feature}"]) for row in pair_rows])),
                "synthetic": float(np.median([float(row[f"synthetic_{feature}"]) for row in pair_rows])),
            }
            for feature in ("contrast", "energy")
        },
        "synthetics_below_original": {
            feature: int(sum(float(row[f"synthetic_{feature}"]) < float(row[f"original_{feature}"]) for row in pair_rows))
            for feature in ("contrast", "energy")
        },
    }
    (OUTPUT / "resumo.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
