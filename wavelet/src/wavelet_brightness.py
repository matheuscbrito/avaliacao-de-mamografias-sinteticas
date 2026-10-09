"""Como a wavelet responde ao brilho?

Duas partes:

1. Varredura controlada nas originais aceitas: brilho (soma), contraste
   (multiplica em torno da mediana do pulmão) e lavagem (mistura o pulmão com o
   P99 real). Para cada nível mede energia e entropia de detalhe e a média da
   banda de aproximação (LL), em duas versões:
   - "pipeline": com o corte em P1–P99 usado no MVP;
   - "sem corte": mesma normalização linear, sem cortar nada.
   A diferença entre as duas mostra quanto do efeito vem do corte, não da wavelet.
2. Dados reais: a média da LL no pulmão para original, Dgen e Dsr.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pywt

from .compare_generations import original_paths
from .density_features import reference_interval
from .load_data import CasePaths, load_image
from .run_pipeline import write_csv
from .validate_cases import RejectedCase, validate_case
from .wavelet_features_fullimage import _block_mean, wavelet_energy_entropy_fullimage

VERSIONS = {"pipeline": "#4C78A8", "sem corte": "#F58518"}


def ll_mean(image, mask, clip_low, clip_high, wavelet, level, min_tissue, clip=True) -> float:
    """Média da banda LL nos blocos de pulmão, de volta à escala 0–1 da normalização."""
    normalised = (np.clip(image, clip_low, clip_high) if clip else image) - clip_low
    normalised = normalised / (clip_high - clip_low)
    approximation = pywt.wavedec2(normalised, wavelet=wavelet, level=level, mode="periodization")[0]
    selection = _block_mean(mask, 2**level)[: approximation.shape[0], : approximation.shape[1]] >= min_tissue
    # cada nível de Haar multiplica a média local por 2
    return float(approximation[selection].mean() / 2**level)


def measure(image, mask, interval, args) -> dict[str, float]:
    width = interval["p99"] - interval["p1"]
    ranges = {"pipeline": (interval["p1"], interval["p99"]), "sem corte": (interval["p1"] - 100 * width, interval["p99"] + 100 * width)}
    out = {}
    for version, (low, high) in ranges.items():
        features = wavelet_energy_entropy_fullimage(image, mask, low, high, args.wavelet, args.level, args.min_tissue)
        out[f"{version}|energy"] = features["energy"]
        out[f"{version}|entropy"] = features["entropy"]
        # a LL "sem corte" usa a mesma escala do pipeline para ficar legível (0 = P1, 1 = P99)
        out[f"{version}|ll"] = ll_mean(image, mask, interval["p1"], interval["p99"], args.wavelet, args.level, args.min_tissue, clip=version == "pipeline")
    return out


def sweeps(interval):
    width = interval["p99"] - interval["p1"]

    def offset(k):
        return lambda image, mask: image + k * width

    def gain(a):
        def apply(image, mask):
            center = np.median(image[mask])
            return center + a * (image - center)
        return apply

    def washout(w):
        def apply(image, mask):
            out = image.copy()
            out[mask] = (1 - w) * image[mask] + w * interval["p99"]
            return out
        return apply

    return {
        "brilho": ("deslocamento (fração da faixa P1–P99)", [(k, offset(k)) for k in np.round(np.linspace(-0.3, 0.3, 13), 2)]),
        "contraste": ("fator de contraste", [(a, gain(a)) for a in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)]),
        "lavagem": ("fração misturada com o P99", [(w, washout(w)) for w in (0.0, 0.25, 0.5, 0.75)]),
    }


def save_curves(rows, output: Path) -> None:
    families = list(dict.fromkeys(row["family"] for row in rows))
    metrics = (("energy", "Energia de detalhe\n(% da original)"), ("entropy", "Entropia de detalhe\n(% da original)"), ("ll", "Média da LL\n(0 = P1, 1 = P99)"))
    fig, axes = plt.subplots(len(metrics), len(families), figsize=(4.4 * len(families), 3.1 * len(metrics)), sharex="col")
    for column, family in enumerate(families):
        subset = [row for row in rows if row["family"] == family]
        xs = sorted({row["x"] for row in subset})
        for line, (metric, label) in enumerate(metrics):
            axis = axes[line, column]
            for version, color in VERSIONS.items():
                key = f"{version}|{metric}" if metric == "ll" else f"{version}|{metric}_pct"
                values = [[row[key] for row in subset if row["x"] == x] for x in xs]
                median = [np.median(v) for v in values]
                low, high = zip(*(np.percentile(v, (25, 75)) for v in values))
                axis.plot(xs, median, "o-", color=color, label=version, markersize=3)
                axis.fill_between(xs, low, high, color=color, alpha=.2)
            if metric != "ll":
                axis.axhline(100, color="#888888", linestyle=":", linewidth=1)
            if metric == "entropy":
                # escala mínima de 80–105%: sem isso, variações de 0,1% parecem efeito
                low_lim, high_lim = axis.get_ylim()
                axis.set_ylim(min(low_lim, 80), max(high_lim, 105))
            if family == "contraste" and metric == "energy":
                grid = np.linspace(min(xs), max(xs), 50)
                axis.plot(grid, 100 * grid**2, "--", color="#555555", linewidth=1, label="teoria: fator²")
            axis.set_ylabel(label if column == 0 else "")
            if line == 0:
                axis.set_title(family.capitalize(), fontsize=12)
            if line == len(metrics) - 1:
                axis.set_xlabel(subset[0]["x_label"])
            axis.legend(fontsize=7)
    fig.suptitle("Como a wavelet responde a brilho, contraste e lavagem (mediana e IQR de 37 originais)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_real(real_rows, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    x = [row["original_ll"] for row in real_rows]
    for axis, (metric, title) in zip(axes, (("ll", "Média da LL: vê o brilho"), ("energy", "Energia de detalhe: não vê"))):
        x = [row[f"original_{metric}"] for row in real_rows]
        values = list(x)
        for generation, color in (("dgen", "#E45756"), ("dsr", "#4C78A8")):
            y = [row[f"{generation}_{metric}"] for row in real_rows]
            values += y
            axis.scatter(x, y, color=color, alpha=.8, label=generation.capitalize())
        for row in real_rows:
            if row["highlight"]:
                axis.annotate(row["short_id"], (row[f"original_{metric}"], row[f"dgen_{metric}"]), fontsize=7, xytext=(4, 2), textcoords="offset points")
        lower, upper = min(values), max(values)
        axis.plot([lower, upper], [lower, upper], "--", color="#888888", label="sintética = original")
        axis.set(title=title, xlabel="Original", ylabel="Sintética")
        axis.legend(fontsize=8)
    fig.suptitle("Dados reais: cada ponto é um corte. Os cortes quebrados do Dgen (078, 081) só fogem da diagonal na LL", fontsize=10)
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dgen-dir", type=Path, required=True)
    parser.add_argument("--dsr-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wavelet", type=str, default="haar")
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--min-tissue", type=float, default=0.9)
    parser.add_argument("--erosion-radius", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = []
    for case_id in sorted(original_paths(args.dgen_dir)):
        paths = CasePaths(case_id, args.dgen_dir / f"{case_id}.tiff", args.dgen_dir / f"{case_id}_generate.tiff", args.dgen_dir / f"{case_id}_mask.tiff")
        result = validate_case(paths, erosion_radius=args.erosion_radius)
        if not isinstance(result, RejectedCase):
            cases.append((result, load_image(args.dsr_dir / f"{case_id}_generate.tiff")))
    interval = reference_interval([(case.original, case.lung_mask) for case, _ in cases])

    rows = []
    for case, _ in cases:
        base = measure(case.original, case.lung_mask, interval, args)
        for family, (x_label, items) in sweeps(interval).items():
            for x, apply in items:
                values = measure(apply(case.original, case.lung_mask), case.lung_mask, interval, args)
                row = {"case_id": case.paths.case_id, "family": family, "x": float(x), "x_label": x_label, **values}
                for version in VERSIONS:
                    for metric in ("energy", "entropy"):
                        row[f"{version}|{metric}_pct"] = 100 * values[f"{version}|{metric}"] / base[f"{version}|{metric}"]
                rows.append(row)

    real_rows = []
    for case, dsr in cases:
        row = {"case_id": case.paths.case_id, "short_id": case.paths.case_id[-3:]}
        for name, image in (("original", case.original), ("dgen", case.synthetic), ("dsr", dsr)):
            values = measure(image, case.lung_mask, interval, args)
            row[f"{name}_ll"] = values["pipeline|ll"]
            row[f"{name}_energy"] = values["pipeline|energy"]
        row["highlight"] = row["dgen_ll"] - row["original_ll"] > 0.3
        real_rows.append(row)

    figures = args.output_dir / "figuras"
    figures.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "varredura_por_caso.csv", rows)
    write_csv(args.output_dir / "ll_dados_reais.csv", real_rows)
    summary = {}
    for family in dict.fromkeys(row["family"] for row in rows):
        summary[family] = {}
        for x in sorted({row["x"] for row in rows if row["family"] == family}):
            subset = [row for row in rows if row["family"] == family and row["x"] == x]
            summary[family][str(x)] = {
                key: float(np.median([row[key] for row in subset]))
                for key in (f"{v}|{m}" for v in VERSIONS for m in ("energy_pct", "entropy_pct", "ll"))
            }
    summary["real_ll_shift_median"] = {g: float(np.median([row[f"{g}_ll"] - row["original_ll"] for row in real_rows])) for g in ("dgen", "dsr")}
    summary["real_highlight"] = [(row["short_id"], round(row["dgen_ll"] - row["original_ll"], 3)) for row in real_rows if row["highlight"]]
    (args.output_dir / "resumo.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    save_curves(rows, figures / "wavelet_resposta_brilho.png")
    save_real(real_rows, figures / "wavelet_ll_dados_reais.png")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
