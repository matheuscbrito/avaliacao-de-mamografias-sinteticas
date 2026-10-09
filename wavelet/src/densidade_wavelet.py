"""Densidade + wavelet: quando o pulmão clareia, o que a wavelet denuncia?

A métrica de densidade diz *se* o brilho do pulmão mudou. A wavelet diz o que
aconteceu com a textura junto com o brilho. Este script:

1. aplica seis jeitos diferentes de clarear o pulmão às originais. Para a
   densidade todos parecem iguais ("clareou"); a ideia é ver se a wavelet os
   separa;
2. lê as imagens geradas do dataset padrão com os dois vereditos lado a lado,
   mais os cortes do dataset de exemplos (Dgen) que a densidade sinalizou.

Os limites do veredito da wavelet (80–125% de energia, razão 0,9–1,1) são uma
escolha exploratória, não um valor validado.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from .density_features import density_features, density_limits, density_shift, density_status, reference_interval
from .run_pipeline import write_csv
from .wavelet_diagnostico import cell_color, cell_text, load_cases, raw_readings, relative

DENSITY_TEXT = {"ok": "normal", "brighter": "clareou", "darker": "escureceu", "too_bright": "claro demais", "too_dark": "escuro demais"}
COLUMNS = (("density", "Densidade\ndiz"), ("ll", "LL\n(brilho)"), ("fine", "Energia\nfina"), ("coarse", "Energia\ngrossa"), ("ratio", "Fina ÷\ngrossa"), ("wavelet", "Wavelet\ndiz"))


def short_name(case_id: str) -> str:
    """Corte + fase (ex.: "078, 40.0A"): o número do corte se repete entre séries."""
    phase = re.search(r"Gated ([\d.]+A)", case_id)
    return f"{case_id[-3:]}, {phase.group(1)}" if phase else case_id[-3:]


def wavelet_verdict(values: dict[str, float]) -> str:
    """Lê a assinatura: quanta textura sobrou e em que escala ela foi perdida."""
    fine, coarse, ratio = values["fine"], values["coarse"], values["ratio"]
    if 80 <= fine <= 125 and 80 <= coarse <= 125:
        return "textura preservada"
    if ratio < 0.9:
        return "borrou"
    if ratio > 1.1:
        return "ruído" if fine > 100 else "estourou"
    return "perdeu contraste" if fine < 100 else "ganhou contraste"


def brightening(interval):
    """Seis jeitos de clarear o pulmão em quantidade parecida."""
    width = interval["p99"] - interval["p1"]

    def lung_only(function):
        def apply(image, mask):
            out = image.copy()
            out[mask] = function(image, mask)[mask]
            return out
        return apply

    def gamma(image, mask):
        low, high = image.min(), image.max()
        return low + (high - low) * ((image - low) / (high - low)) ** 0.6

    def fill_floor(image, mask):
        return np.maximum(image, np.median(image[mask]))

    def washout_half(image, mask):
        # metade de baixo do pulmão vira quase branca, como nas manchas do Dgen
        rows = np.where(mask)[0]
        out = image.copy()
        region = mask & (np.arange(image.shape[0])[:, None] > np.median(rows))
        out[region] = 0.4 * image[region] + 0.6 * interval["p99"]
        return out

    return [
        ("Só brilho (+25%)", lambda image, mask: image + 0.25 * width),
        ("Lavagem (40% branco)", lung_only(lambda image, mask: 0.6 * image + 0.4 * interval["p99"])),
        ("Gama (0,6)", gamma),
        ("Fundo preenchido (até a mediana)", lung_only(fill_floor)),
        ("Brilho + borrado (+25%, σ=1)", lambda image, mask: gaussian_filter(image, 1) + 0.25 * width),
        ("Lavagem só em metade do pulmão", washout_half),
    ]


def save_table(rows, output: Path, title: str) -> None:
    numeric = [key for key, _ in COLUMNS if key not in ("density", "wavelet")]
    colors = np.zeros((len(rows), len(COLUMNS)))
    for row, (_, values) in enumerate(rows):
        for column, (key, _) in enumerate(COLUMNS):
            colors[row, column] = cell_color(key, values[key]) if key in numeric else 0
    fig, axis = plt.subplots(figsize=(13.5, 0.55 * len(rows) + 1.7))
    axis.imshow(colors, cmap="RdBu_r", vmin=-1.6, vmax=1.6, aspect="auto")
    for row, (_, values) in enumerate(rows):
        for column, (key, _) in enumerate(COLUMNS):
            text = cell_text(key, values[key]) if key in numeric else values[key]
            weight = "bold" if key in ("density", "wavelet") else "normal"
            axis.text(column, row, text.replace(" (", "\n("), ha="center", va="center", fontsize=9, fontweight=weight)
    axis.set_xticks(range(len(COLUMNS)), [label for _, label in COLUMNS], fontsize=10)
    axis.xaxis.tick_top()
    axis.set_yticks(range(len(rows)), [label for label, _ in rows], fontsize=10)
    axis.set_xticks(np.arange(-0.5, len(COLUMNS)), minor=True)
    axis.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    axis.grid(which="minor", color="white", linewidth=2)
    axis.tick_params(which="minor", length=0)
    axis.set_title(title, fontsize=11, pad=42)
    fig.text(0.5, 0.01, "LL: quanto o brilho andou (fração da faixa P1–P99). Energias: % da original. Fina ÷ grossa < 1 = perdeu mais o fino.", ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_examples(case, items, interval, output: Path) -> None:
    rows, cols = np.where(case.lung_mask)
    crop = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
    panels = [("Original", case.original)] + [(name, apply(case.original, case.lung_mask)) for name, apply in items]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.9 * len(panels), 3.2))
    for axis, (title, image) in zip(axes, panels):
        axis.imshow(image[crop], cmap="gray", vmin=interval["p1"], vmax=interval["p99"])
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def read_case(image, case, base_density, base_wavelet, interval, limits, args) -> dict:
    density = density_features(image, case.lung_mask, interval)
    values = relative(raw_readings(image, case.lung_mask, interval, args), base_wavelet)
    values["density"] = DENSITY_TEXT[density_status(density, base_density, limits)]
    values["median_shift_iqr"] = density_shift(density, base_density, limits)["median_shift_iqr"]
    values["wavelet"] = wavelet_verdict(values)
    return values


def summarise(readings: list[dict]) -> dict:
    """Mediana das leituras numéricas e o veredito mais comum de cada eixo."""
    out = {key: float(np.median([item[key] for item in readings])) for key in ("ll", "fine", "coarse", "ratio", "median_shift_iqr")}
    for key in ("density", "wavelet"):
        label, count = Counter(item[key] for item in readings).most_common(1)[0]
        out[key] = label if count == len(readings) else f"{label} ({count}/{len(readings)})"
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="dataset padrão (Dsr)")
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--examples-dir", type=Path, help="dataset de exemplos quebrados (Dgen)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wavelet", type=str, default="haar")
    parser.add_argument("--min-tissue", type=float, default=0.9)
    parser.add_argument("--erosion-radius", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_cases(args.data_dir, args.mask_dir, args.erosion_radius)
    interval = reference_interval([(case.original, case.lung_mask) for case in cases])
    base_density = {case.paths.case_id: density_features(case.original, case.lung_mask, interval) for case in cases}
    limits = density_limits(list(base_density.values()))
    base_wavelet = {case.paths.case_id: raw_readings(case.original, case.lung_mask, interval, args) for case in cases}

    def read(image, case):
        return read_case(image, case, base_density[case.paths.case_id], base_wavelet[case.paths.case_id], interval, limits, args)

    # parte 1: seis jeitos de clarear
    items = brightening(interval)
    per_case, simulated = [], []
    for name, apply in items:
        readings = [read(apply(case.original, case.lung_mask), case) for case in cases]
        per_case += [{"source": "simulado", "label": name, "case_id": case.paths.case_id, **values} for case, values in zip(cases, readings)]
        simulated.append((name, summarise(readings)))

    # parte 2: imagens geradas
    generated = {case.paths.case_id: read(case.synthetic, case) for case in cases}
    per_case += [{"source": "dataset padrão", "label": "gerada", "case_id": case_id, **values} for case_id, values in generated.items()]
    crosstab = Counter((values["density"], values["wavelet"]) for values in generated.values())
    real_rows = [(f"Padrão: {density} + {wavelet} ({count})", summarise([v for v in generated.values() if v["density"] == density and v["wavelet"] == wavelet])) for (density, wavelet), count in crosstab.most_common()]

    if args.examples_dir:
        example_cases = {case.paths.case_id: case for case in load_cases(args.examples_dir, args.mask_dir, args.erosion_radius)}
        flagged = {case_id: read(case.synthetic, case) for case_id, case in example_cases.items()}
        flagged = {case_id: values for case_id, values in flagged.items() if values["density"] != "normal"}
        per_case += [{"source": "exemplo", "label": "gerada", "case_id": case_id, **values} for case_id, values in flagged.items()]
        real_rows += [(f"Exemplo {short_name(case_id)}", values) for case_id, values in sorted(flagged.items(), key=lambda item: -item[1]["ll"]) if values["density"] in ("claro demais", "clareou")]

    figures = args.output_dir / "figuras"
    figures.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "leituras_por_caso.csv", per_case)
    (args.output_dir / "resumo.json").write_text(json.dumps({
        "n_originais": len(cases), "limites_densidade": limits,
        "simulado": dict(simulated),
        "dataset_padrao_cruzamento": {f"{d} + {w}": n for (d, w), n in crosstab.most_common()},
        "linhas_reais": dict(real_rows),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    save_table(simulated, figures / "clarear_seis_jeitos.png", "Seis jeitos de clarear o pulmão: o que a densidade e a wavelet dizem (mediana de 37 cortes)")
    save_table(real_rows, figures / "dados_reais.png", "Imagens geradas: veredito da densidade e da wavelet lado a lado")
    middle = cases[len(cases) // 2]
    save_examples(middle, items, interval, figures / "clarear_exemplos.png")
    for label, values in simulated + real_rows:
        print(f"{label:52s} dens={values['density']:22s} LL={values['ll']:+.2f} fina={values['fine']:.0f}% grossa={values['coarse']:.0f}% razão={values['ratio']:.2f} shift={values['median_shift_iqr']:+.1f} → {values['wavelet']}")


if __name__ == "__main__":
    main()
