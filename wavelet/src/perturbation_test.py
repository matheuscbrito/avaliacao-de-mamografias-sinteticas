"""Teste de perturbação controlada: qual eixo do framework reage a qual defeito?

Só usa as **originais** aceitas. Cada uma recebe defeitos conhecidos — brilho,
contraste, gama, lavagem do pulmão, saturação, suavização e ruído — e todos os
eixos são medidos antes e depois:

- densidade (`density_features.py`): mediana do pulmão e frações fora da faixa real;
- wavelet (imagem inteira): energia e entropia das sub-bandas de detalhe;
- GLCM (`../glcm/src/glcm_features.py`): contraste e energia;
- NEx (`nyquist_excess.py`): excesso perto de Nyquist, imagem inteira.

A mudança de cada métrica é dividida pelo IQR dessa métrica entre as originais
sem defeito. Assim os eixos ficam na mesma régua: |efeito| > 1 quer dizer que o
defeito mexeu na métrica mais do que um pulmão real difere de outro.

Como o defeito é conhecido, o resultado diz se cada eixo mede o que promete —
por exemplo, se a densidade pega brilho e se a textura ignora brilho, mas pega
suavização.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from .density_features import DENSITY_STATUSES, density_features, density_limits, density_status, reference_interval
from .load_data import discover_cases
from .run_pipeline import write_csv
from .validate_cases import RejectedCase, validate_case
from .wavelet_features_fullimage import wavelet_energy_entropy_fullimage
from .nyquist_excess import nyquist_excess

GLCM_FEATURES_PATH = Path(__file__).resolve().parents[2] / "glcm" / "src" / "glcm_features.py"
METRICS = (
    ("density", "lung_median"), ("density", "frac_bright"), ("density", "frac_dark"),
    ("wavelet", "energy"), ("wavelet", "entropy"),
    ("glcm", "contrast"), ("glcm", "energy"),
    ("nex", "score"),
)


def load_glcm():
    """Importa a GLCM do MVP irmão sem copiar o código (os dois pacotes se chamam `src`)."""
    spec = importlib.util.spec_from_file_location("glcm_features", GLCM_FEATURES_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.glcm_contrast_energy


def perturbations(interval: dict[str, float]) -> list[tuple[str, str, object]]:
    """(família, nível, função imagem, máscara → imagem). Intensidades em frações da faixa P1–P99."""
    width = interval["p99"] - interval["p1"]
    rng = np.random.default_rng(0)

    def offset(k):
        return lambda image, mask: image + k * width

    def gain(a):
        def apply(image, mask):
            center = np.median(image[mask])
            return center + a * (image - center)
        return apply

    def gamma(g):
        def apply(image, mask):
            low, high = image.min(), image.max()
            return low + (high - low) * ((image - low) / (high - low)) ** g
        return apply

    def washout(w):
        # mistura o pulmão com um valor claro (o P99 real): sobe o brilho e apaga a textura
        def apply(image, mask):
            out = image.copy()
            out[mask] = (1 - w) * image[mask] + w * interval["p99"]
            return out
        return apply

    def clip_top(q):
        # vasos e estruturas claras viram um platô: saturação no topo
        def apply(image, mask):
            out = image.copy()
            out[mask] = np.minimum(image[mask], np.percentile(image[mask], q))
            return out
        return apply

    def clip_bottom(q):
        # parênquima escuro vira um platô: saturação no fundo
        def apply(image, mask):
            out = image.copy()
            out[mask] = np.maximum(image[mask], np.percentile(image[mask], q))
            return out
        return apply

    def blur(sigma):
        return lambda image, mask: gaussian_filter(image, sigma)

    def noise(k):
        return lambda image, mask: image + rng.normal(0, k * width, image.shape)

    return [
        ("brilho", "-20%", offset(-0.2)), ("brilho", "-10%", offset(-0.1)),
        ("brilho", "+10%", offset(0.1)), ("brilho", "+20%", offset(0.2)),
        ("contraste", "×0,5", gain(0.5)), ("contraste", "×1,5", gain(1.5)),
        ("gama", "0,7", gamma(0.7)), ("gama", "1,4", gamma(1.4)),
        ("lavagem", "25%", washout(0.25)), ("lavagem", "50%", washout(0.5)), ("lavagem", "75%", washout(0.75)),
        ("saturação topo", "P90", clip_top(90)), ("saturação topo", "P75", clip_top(75)),
        ("saturação fundo", "P10", clip_bottom(10)), ("saturação fundo", "P25", clip_bottom(25)),
        ("suavização", "σ=1", blur(1)), ("suavização", "σ=2", blur(2)),
        ("ruído", "2%", noise(0.02)), ("ruído", "5%", noise(0.05)),
    ]


def measure(image, mask, interval, args, glcm) -> dict[str, float]:
    wavelet = wavelet_energy_entropy_fullimage(image, mask, interval["p1"], interval["p99"], args.wavelet, args.level, args.min_tissue)
    texture = glcm(image, mask, interval["p1"], interval["p99"], args.glcm_levels)
    return {
        **{f"density_{name}": value for name, value in density_features(image, mask, interval).items()},
        "wavelet_energy": wavelet["energy"], "wavelet_entropy": wavelet["entropy"],
        "glcm_contrast": texture["contrast"], "glcm_energy": texture["energy"],
        "nex_score": nyquist_excess(image),
    }


def save_heatmap(effects: dict[str, dict[str, float]], output: Path) -> None:
    labels = list(effects)
    columns = [f"{axis}_{name}" for axis, name in METRICS]
    matrix = np.array([[effects[label][column] for column in columns] for label in labels])
    fig, axis = plt.subplots(figsize=(10, 0.42 * len(labels) + 1.8))
    image = axis.imshow(np.clip(matrix, -5, 5), cmap="RdBu_r", vmin=-5, vmax=5, aspect="auto")
    for (row, column), value in np.ndenumerate(matrix):
        axis.text(column, row, f"{value:+.1f}", ha="center", va="center", fontsize=7, color="white" if abs(value) > 3 else "black")
    axis.set_xticks(range(len(columns)), [column.replace("_", "\n", 1) for column in columns], fontsize=8)
    axis.set_yticks(range(len(labels)), labels, fontsize=8)
    for boundary in (2.5, 4.5, 6.5):
        axis.axvline(boundary, color="#333333", linewidth=1)
    fig.colorbar(image, ax=axis, label="mudança mediana / IQR entre originais (cortado em ±5)")
    axis.set_title("Quanto cada defeito mexe em cada métrica (|valor| > 1 = mais que a variação entre pulmões reais)")
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def save_examples(case, interval, items, output: Path) -> None:
    image, mask = case.original, case.lung_mask
    rows_, cols_ = np.where(mask)
    crop = (slice(rows_.min(), rows_.max() + 1), slice(cols_.min(), cols_.max() + 1))
    panels = [("original", image)] + [(f"{family} {level}", apply(image, mask)) for family, level, apply in items]
    columns = 5
    fig, axes = plt.subplots(int(np.ceil(len(panels) / columns)), columns, figsize=(columns * 2.6, 2.6 * np.ceil(len(panels) / columns)))
    for axis in axes.flat:
        axis.axis("off")
    for axis, (title, panel) in zip(axes.flat, panels):
        # mesma janela de exibição em todos, para o brilho ser comparável
        axis.imshow(panel[crop], cmap="gray", vmin=interval["p1"], vmax=interval["p99"])
        axis.set_title(title, fontsize=8)
    fig.suptitle(f"Defeitos aplicados a {case.paths.case_id} (janela fixa P1–P99 do pulmão real)", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wavelet", type=str, default="haar")
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--min-tissue", type=float, default=0.9)
    parser.add_argument("--erosion-radius", type=int, default=2)
    parser.add_argument("--glcm-levels", type=int, default=32)
    parser.add_argument("--median-threshold", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    glcm = load_glcm()
    valid = [result for result in (validate_case(paths, erosion_radius=args.erosion_radius) for paths in discover_cases(args.data_dir)) if not isinstance(result, RejectedCase)]
    interval = reference_interval([(case.original, case.lung_mask) for case in valid])
    limits = density_limits([density_features(case.original, case.lung_mask, interval) for case in valid])
    items = perturbations(interval)

    baselines = {case.paths.case_id: measure(case.original, case.lung_mask, interval, args, glcm) for case in valid}
    columns = [f"{axis}_{name}" for axis, name in METRICS]
    iqr = {column: float(np.subtract(*np.percentile([values[column] for values in baselines.values()], (75, 25)))) for column in columns}

    rows: list[dict] = []
    for case in valid:
        base = baselines[case.paths.case_id]
        base_density = {name: base[f"density_{name}"] for name in ("lung_median", "frac_bright", "frac_dark")}
        for family, level, apply in items:
            perturbed = apply(case.original, case.lung_mask)
            values = measure(perturbed, case.lung_mask, interval, args, glcm)
            row = {"case_id": case.paths.case_id, "family": family, "level": level}
            for column in columns:
                row[column] = values[column]
                row[f"effect_{column}"] = (values[column] - base[column]) / max(iqr[column], 1e-12)
            row["density_status"] = density_status({name: values[f"density_{name}"] for name in base_density}, base_density, limits, args.median_threshold)
            rows.append(row)

    labels = [f"{family} {level}" for family, level, _ in items]
    effects, detection = {}, {}
    for label, (family, level, _) in zip(labels, items):
        subset = [row for row in rows if row["family"] == family and row["level"] == level]
        effects[label] = {column: float(np.median([row[f"effect_{column}"] for row in subset])) for column in columns}
        detection[label] = {status: sum(row["density_status"] == status for row in subset) / len(subset) for status in DENSITY_STATUSES}

    figures = args.output_dir / "figuras"
    figures.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "perturbacoes_por_caso.csv", rows)
    write_csv(args.output_dir / "sensibilidade.csv", [{"perturbation": label, **values, **{f"density_{status}": share for status, share in detection[label].items()}} for label, values in effects.items()])
    summary = {
        "n_originals": len(valid), "reference_interval": interval, "density_limits": limits,
        "baseline_iqr": iqr, "median_effect_iqr": effects, "density_status_share": detection,
        "wavelet": args.wavelet, "level": args.level, "glcm_levels": args.glcm_levels,
    }
    (args.output_dir / "resumo.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    save_heatmap(effects, figures / "sensibilidade.png")
    save_examples(valid[len(valid) // 2], interval, items, figures / "exemplos_perturbacao.png")
    print(json.dumps({"n_originals": len(valid), "density_status_share": detection}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
