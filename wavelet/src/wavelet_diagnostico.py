"""O que a wavelet diz sobre o tipo de defeito?

A métrica de brilho avisa *que* algo mudou. A wavelet ajuda a dizer *o quê*,
com quatro leituras da sintética em relação à sua original:

- LL: média da banda de aproximação (o "borrão" da imagem) — o brilho mudou?
- energia fina: detalhe na escala de 2 px — a textura mais fina continua lá?
- energia grossa: detalhe na escala de 4 px — e a textura um pouco maior?
- fina ÷ grossa: abaixo de 1 = perdeu mais detalhe fino (borramento); perto de
  1 = perdeu igual nas duas escalas (contraste); acima = ganhou fino (ruído);
- entropia: a textura está espalhada ou virou platôs?

Parte 1 aplica defeitos conhecidos às originais e anota a assinatura de cada
um. Parte 2 lê imagens geradas com as mesmas quatro leituras: o dataset
padrão (Dsr) e alguns exemplos quebrados do Dgen (branco, borrado).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.ndimage import gaussian_filter

from .compare_generations import original_paths
from .density_features import reference_interval
from .load_data import CasePaths, load_image
from .run_pipeline import write_csv
from .validate_cases import RejectedCase, validate_case
from .wavelet_features_fullimage import _block_mean, wavelet_energy_entropy_fullimage

LEVEL = 2
READINGS = (("ll", "LL\n(brilho)"), ("fine", "Energia fina\n(2 px)"), ("coarse", "Energia grossa\n(4 px)"), ("ratio", "Fina ÷ grossa\n(tipo de perda)"), ("entropy", "Entropia"))


def raw_readings(image, mask, interval, args) -> dict[str, float]:
    low, high = interval["p1"], interval["p99"]
    features = wavelet_energy_entropy_fullimage(image, mask, low, high, args.wavelet, LEVEL, args.min_tissue)
    approximation = pywt.wavedec2((np.clip(image, low, high) - low) / (high - low), wavelet=args.wavelet, level=LEVEL, mode="periodization")[0]
    selection = _block_mean(mask, 2**LEVEL)[: approximation.shape[0], : approximation.shape[1]] >= args.min_tissue
    # as chaves L1 e L2 seguem a ordem do wavedec2: L1 é a escala grossa, L2 a fina
    return {
        "ll": float(approximation[selection].mean() / 2**LEVEL),
        "fine": float(np.mean([features[f"energy_L2_{band}"] for band in ("LH", "HL", "HH")])),
        "coarse": float(np.mean([features[f"energy_L1_{band}"] for band in ("LH", "HL", "HH")])),
        "entropy": features["entropy"],
    }


def relative(values: dict[str, float], base: dict[str, float]) -> dict[str, float]:
    """LL como deslocamento (fração da faixa P1–P99); razão fina/grossa pura; o resto em % da original."""
    out = {"ll": values["ll"] - base["ll"], **{key: 100 * values[key] / base[key] for key in ("fine", "coarse", "entropy")}}
    out["ratio"] = out["fine"] / out["coarse"]
    return out


def defects(interval):
    width = interval["p99"] - interval["p1"]
    rng = np.random.default_rng(0)

    def lung_only(function):
        def apply(image, mask):
            out = image.copy()
            out[mask] = function(image, mask)[mask]
            return out
        return apply

    def gain(a):
        return lambda image, mask: np.median(image[mask]) + a * (image - np.median(image[mask]))

    return [
        ("Simulado: mais claro (+20%)", lambda image, mask: image + 0.2 * width),
        ("Simulado: mais escuro (−20%)", lambda image, mask: image - 0.2 * width),
        ("Simulado: menos contraste (×0,5)", gain(0.5)),
        ("Simulado: mais contraste (×1,5)", gain(1.5)),
        ("Simulado: lavagem (50% branco)", lung_only(lambda image, mask: 0.5 * image + 0.5 * interval["p99"])),
        ("Simulado: estourado (topo em P75)", lung_only(lambda image, mask: np.minimum(image, np.percentile(image[mask], 75)))),
        ("Simulado: borrado (σ=1)", lambda image, mask: gaussian_filter(image, 1)),
        ("Simulado: ruído (5%)", lambda image, mask: image + rng.normal(0, 0.05 * width, image.shape)),
    ]


def cell_color(key: str, value: float) -> float:
    """Valor entre −1 e 1 para a cor: LL em passos de 0,2 da faixa; o resto em log2 da razão."""
    if key == "ll":
        return float(np.clip(value / 0.2, -1, 1))
    if key == "ratio":
        return float(np.clip(2.5 * np.log2(max(value, 1e-6)), -1, 1))
    return float(np.clip(np.log2(max(value, 1e-6) / 100), -1, 1))


def cell_text(key: str, value: float) -> str:
    if key == "ll":
        return f"{value:+.2f}".replace(".", ",")
    if key == "ratio":
        return f"{value:.2f}".replace(".", ",")
    return f"{value:.0f}%"


def save_table(rows: list[tuple[str, dict[str, float]]], separator: int, output: Path) -> None:
    keys = [key for key, _ in READINGS]
    colors = np.array([[cell_color(key, values[key]) for key in keys] for _, values in rows])
    fig, axis = plt.subplots(figsize=(9.5, 0.5 * len(rows) + 1.6))
    axis.imshow(colors, cmap="RdBu_r", vmin=-1.6, vmax=1.6, aspect="auto")
    for row, (_, values) in enumerate(rows):
        for column, key in enumerate(keys):
            axis.text(column, row, cell_text(key, values[key]), ha="center", va="center", fontsize=10)
    axis.set_xticks(range(len(keys)), [label for _, label in READINGS], fontsize=10)
    axis.xaxis.tick_top()
    axis.set_yticks(range(len(rows)), [label for label, _ in rows], fontsize=10)
    axis.axhline(separator - 0.5, color="#222222", linewidth=2)
    axis.set_xticks(np.arange(-0.5, len(keys)), minor=True)
    axis.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    axis.grid(which="minor", color="white", linewidth=2)
    axis.tick_params(which="minor", length=0)
    fig.text(0.5, 0.01, "LL: quanto o brilho andou (fração da faixa P1–P99). Energias e entropia: % da original.\nFina ÷ grossa: < 1 perdeu mais detalhe fino. Vermelho = subiu, azul = caiu.", ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0.03, 0.03, 1, 1))
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_examples(examples, interval, output: Path) -> None:
    fig, axes = plt.subplots(2, len(examples), figsize=(4 * len(examples), 7.2))
    for column, (title, case, synthetic) in enumerate(examples):
        rows, cols = np.where(case.lung_mask)
        crop = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
        for row, (name, image) in enumerate((("original", case.original), ("gerada", synthetic))):
            axis = axes[row, column]
            axis.imshow(image[crop], cmap="gray", vmin=interval["p1"], vmax=interval["p99"])
            axis.set_title(f"{title}\n{name}" if row == 0 else name, fontsize=10)
            axis.axis("off")
    fig.suptitle("Exemplos lidos na tabela (mesma janela de cinza em todas)", fontsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_cases(data_dir: Path, mask_dir: Path, erosion_radius: int):
    """Originais + geradas de `data_dir`; as máscaras vêm de `mask_dir` (o Dsr não traz máscaras)."""
    cases = []
    for case_id in sorted(original_paths(data_dir)):
        paths = CasePaths(case_id, data_dir / f"{case_id}.tiff", data_dir / f"{case_id}_generate.tiff", mask_dir / f"{case_id}_mask.tiff")
        result = validate_case(paths, erosion_radius=erosion_radius)
        if not isinstance(result, RejectedCase):
            cases.append(result)
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="dataset padrão (Dsr)")
    parser.add_argument("--mask-dir", type=Path, required=True, help="pasta com as máscaras *_mask.tiff")
    parser.add_argument("--examples-dir", type=Path, help="dataset de onde sai o exemplo de imagem branca (Dgen)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wavelet", type=str, default="haar")
    parser.add_argument("--min-tissue", type=float, default=0.9)
    parser.add_argument("--erosion-radius", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_cases(args.data_dir, args.mask_dir, args.erosion_radius)
    interval = reference_interval([(case.original, case.lung_mask) for case in cases])
    bases = {case.paths.case_id: raw_readings(case.original, case.lung_mask, interval, args) for case in cases}

    # parte 1: assinatura de cada defeito, mediana entre as originais
    per_case, table = [], []
    for name, apply in defects(interval):
        readings = [relative(raw_readings(apply(case.original, case.lung_mask), case.lung_mask, interval, args), bases[case.paths.case_id]) for case in cases]
        per_case += [{"source": "defeito", "label": name, "case_id": case.paths.case_id, **values} for case, values in zip(cases, readings)]
        table.append((name, {key: float(np.median([values[key] for values in readings])) for key, _ in READINGS}))
    separator = len(table)

    # parte 2: imagens geradas lidas com as mesmas leituras
    generated = {case.paths.case_id: relative(raw_readings(case.synthetic, case.lung_mask, interval, args), bases[case.paths.case_id]) for case in cases}
    per_case += [{"source": "gerada", "label": "dataset padrão", "case_id": case_id, **values} for case_id, values in generated.items()]
    table.append(("Gerada: mediana do dataset", {key: float(np.median([values[key] for values in generated.values()])) for key, _ in READINGS}))
    typical = min(cases, key=lambda case: sum(abs(generated[case.paths.case_id][key] - table[-1][1][key]) / (1 if key == "ll" else 100) for key, _ in READINGS))
    examples = [("Gerada: corte típico", typical, typical.synthetic)]
    # borrado: perde muito mais detalhe fino que grosso, sem mudar o brilho
    blurred = min((case for case in cases if abs(generated[case.paths.case_id]["ll"]) < 0.05), key=lambda case: generated[case.paths.case_id]["ratio"])
    label = f"Gerada: a mais borrada ({blurred.paths.case_id[-3:]})"
    table.append((label, generated[blurred.paths.case_id]))
    examples.append((label, blurred, blurred.synthetic))

    if args.examples_dir:
        broken = {case.paths.case_id: (case, relative(raw_readings(case.synthetic, case.lung_mask, interval, args), bases[case.paths.case_id])) for case in load_cases(args.examples_dir, args.mask_dir, args.erosion_radius)}
        white_id = max(broken, key=lambda case_id: broken[case_id][1]["ll"])
        case, values = broken[white_id]
        label = f"Gerada: exemplo branco ({white_id[-3:]})"
        table.append((label, values))
        per_case.append({"source": "exemplo", "label": label, "case_id": white_id, **values})
        examples.append((label, case, case.synthetic))

    figures = args.output_dir / "figuras"
    figures.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "leituras_por_caso.csv", per_case)
    (args.output_dir / "resumo.json").write_text(json.dumps({"n_originais": len(cases), "tabela": dict(table)}, indent=2, ensure_ascii=False), encoding="utf-8")
    save_table(table, separator, figures / "assinatura_wavelet.png")
    save_examples(examples, interval, figures / "exemplos_lidos.png")
    for label, values in table:
        print(f"{label:30s}", "  ".join(f"{key}={cell_text(key, values[key])}" for key, _ in READINGS))


if __name__ == "__main__":
    main()
