"""Mapa por região: densidade e wavelet em blocos do pulmão, não na média.

Os números globais diluem defeitos locais (ex.: só metade do pulmão lavada).
Aqui a imagem é decomposta uma vez (Haar, 2 níveis, mesmo corte P1–P99 do
pipeline) e cada bloco de `--block` px com pulmão suficiente recebe:

- brilho: deslocamento da média da LL em relação ao mesmo bloco da original;
- energia fina (2 px) e grossa (4 px) em % da original, e a razão entre elas;
- o mesmo veredito de textura de `densidade_wavelet.py`.

Também mede a correlação espacial (Spearman) entre os mapas de energia fina da
original e da gerada: a textura está nos mesmos lugares?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pywt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.stats import spearmanr

from .densidade_wavelet import brightening, short_name, wavelet_verdict
from .density_features import density_features, density_limits, density_status, reference_interval
from .run_pipeline import write_csv
from .wavelet_diagnostico import load_cases
from .wavelet_features_fullimage import _block_mean

VERDICTS = ("textura preservada", "borrou", "perdeu contraste", "estourou", "ruído", "ganhou contraste")
MIN_BLOCKS_FOR_CORRELATION = 10
VERDICT_COLORS = ("#d9d9d9", "#4C78A8", "#9467bd", "#F58518", "#EECA3B", "#e377c2")


def _pool(array: np.ndarray, size: int) -> np.ndarray:
    height, width = (dim - dim % size for dim in array.shape)
    trimmed = array[:height, :width]
    return trimmed.reshape(height // size, size, width // size, size).mean(axis=(1, 3))


def block_maps(image, mask, interval, block: int, wavelet: str, min_tissue: float):
    """Mapas por bloco: média da LL, energia fina e grossa; e quais blocos são pulmão."""
    low, high = interval["p1"], interval["p99"]
    coeffs = pywt.wavedec2((np.clip(image, low, high) - low) / (high - low), wavelet=wavelet, level=2, mode="periodization")
    # wavedec2: [LL2, detalhes do nível 2 (grosso), detalhes do nível 1 (fino)]
    ll = coeffs[0] / 4
    coarse = np.mean([band**2 for band in coeffs[1]], axis=0)
    fine = np.mean([band**2 for band in coeffs[2]], axis=0)
    maps = {"ll": _pool(ll, block // 4), "coarse": _pool(coarse, block // 4), "fine": _pool(fine, block // 2)}
    return maps, _block_mean(mask, block) >= min_tissue


def compare_maps(original, generated, lung, floor):
    """Leituras por bloco da gerada contra a original; `floor` evita razões explosivas em blocos lisos."""
    readings = {
        "ll": generated["ll"] - original["ll"],
        "fine": 100 * generated["fine"] / np.maximum(original["fine"], floor["fine"]),
        "coarse": 100 * generated["coarse"] / np.maximum(original["coarse"], floor["coarse"]),
    }
    readings["ratio"] = readings["fine"] / np.maximum(readings["coarse"], 1e-6)
    verdict = np.full(lung.shape, -1)
    for index in zip(*np.where(lung)):
        verdict[index] = VERDICTS.index(wavelet_verdict({key: readings[key][index] for key in ("fine", "coarse", "ratio")}))
    readings["verdict"] = verdict
    return readings


def summarise(readings, original, generated, lung, bright_shift: float) -> dict[str, float]:
    """Fração da área do pulmão em cada veredito, área com brilho local e correlação espacial.

    `bright_shift` é a mesma tolerância da densidade global: o IQR das medianas
    das originais, na escala da faixa P1–P99.
    """
    total = lung.sum()
    out = {f"area_{name}": float((readings["verdict"][lung] == index).sum() / total) for index, name in enumerate(VERDICTS)}
    out["area_mais_claro"] = float((readings["ll"][lung] > bright_shift).sum() / total)
    out["area_mais_escuro"] = float((readings["ll"][lung] < -bright_shift).sum() / total)
    out["n_blocos"] = float(total)
    # com poucos blocos a correlação não quer dizer nada: fica vazia
    enough = total >= MIN_BLOCKS_FOR_CORRELATION
    out["correlacao_textura"] = float(spearmanr(original["fine"][lung], generated["fine"][lung])[0]) if enough else float("nan")
    out["correlacao_brilho"] = float(spearmanr(original["ll"][lung], generated["ll"][lung])[0]) if enough else float("nan")
    return out


def draw_row(axes, title, case, image, readings, lung, interval, block, stats):
    rows, cols = np.where(case.lung_mask)
    limits = (cols.min() - 4, cols.max() + 4, rows.max() + 4, rows.min() - 4)
    extent = (0, lung.shape[1] * block, lung.shape[0] * block, 0)
    bright = np.where(lung, readings["ll"], np.nan)
    verdict = np.where(lung, readings["verdict"], np.nan).astype(float)
    panels = (
        ("Original", case.original, None),
        (title, image, None),
        (f"Brilho por região\n{stats['area_mais_claro']:.0%} mais claro, {stats['area_mais_escuro']:.0%} mais escuro", image, ("bright", bright)),
        (f"Textura por região\ncorrelação com a original: {stats['correlacao_textura']:.2f}" if np.isfinite(stats["correlacao_textura"]) else f"Textura por região\n(só {stats['n_blocos']:.0f} blocos: sem correlação)", image, ("verdict", verdict)),
    )
    for axis, (label, background, overlay) in zip(axes, panels):
        axis.imshow(background, cmap="gray", vmin=interval["p1"], vmax=interval["p99"], alpha=1 if overlay is None else 0.55)
        if overlay and overlay[0] == "bright":
            axis.imshow(overlay[1], cmap="RdBu_r", vmin=-0.4, vmax=0.4, extent=extent, alpha=0.75, interpolation="nearest")
        elif overlay:
            axis.imshow(overlay[1], cmap=ListedColormap(VERDICT_COLORS), vmin=-0.5, vmax=len(VERDICTS) - 0.5, extent=extent, alpha=0.8, interpolation="nearest")
        axis.contour(case.lung_mask, levels=[0.5], colors="#00a36c", linewidths=0.6)
        axis.set_xlim(limits[0], limits[1])
        axis.set_ylim(limits[2], limits[3])
        axis.set_title(label, fontsize=9)
        axis.axis("off")


def save_figure(entries, interval, block, output: Path, title: str) -> None:
    fig, axes = plt.subplots(len(entries), 4, figsize=(15, 3.6 * len(entries)), squeeze=False)
    for row, entry in zip(axes, entries):
        label, case, image, readings, lung, stats = entry
        draw_row(row, label, case, image, readings, lung, interval, block, stats)
    handles = [Patch(color=color, label=name) for name, color in zip(VERDICTS, VERDICT_COLORS)]
    handles += [Patch(color="#d6604d", label="brilho: mais claro"), Patch(color="#4393c3", label="brilho: mais escuro")]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="dataset padrão (Dsr)")
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--examples-dir", type=Path, help="dataset de exemplos quebrados (Dgen)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block", type=int, default=16, help="lado do bloco em px (múltiplo de 4)")
    parser.add_argument("--wavelet", type=str, default="haar")
    parser.add_argument("--min-tissue", type=float, default=0.9)
    parser.add_argument("--erosion-radius", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.block % 4:
        raise ValueError("--block precisa ser múltiplo de 4")
    cases = load_cases(args.data_dir, args.mask_dir, args.erosion_radius)
    interval = reference_interval([(case.original, case.lung_mask) for case in cases])
    limits = density_limits([density_features(case.original, case.lung_mask, interval) for case in cases])
    originals = {case.paths.case_id: block_maps(case.original, case.lung_mask, interval, args.block, args.wavelet, args.min_tissue) for case in cases}
    # piso das energias: P5 dos blocos de pulmão de todas as originais
    floor = {key: float(np.percentile(np.concatenate([maps[key][lung] for maps, lung in originals.values()]), 5)) for key in ("fine", "coarse")}

    def analyse(case, image):
        maps, lung = originals[case.paths.case_id]
        generated, _ = block_maps(image, case.lung_mask, interval, args.block, args.wavelet, args.min_tissue)
        readings = compare_maps(maps, generated, lung, floor)
        return readings, lung, summarise(readings, maps, generated, lung, limits["lung_median"])

    rows, summary = [], {}
    # parte 1: defeitos simulados (o veredito global vs a área por região)
    selected = [item for item in brightening(interval) if item[0] in ("Só brilho (+25%)", "Lavagem (40% branco)", "Lavagem só em metade do pulmão", "Brilho + borrado (+25%, σ=1)")]
    example_case = cases[len(cases) // 2]
    simulated_entries = []
    for name, apply in selected:
        stats = []
        for case in cases:
            image = apply(case.original, case.lung_mask)
            readings, lung, values = analyse(case, image)
            stats.append(values)
            rows.append({"source": "simulado", "label": name, "case_id": case.paths.case_id, **values})
            if case is example_case:
                simulated_entries.append((name, case, image, readings, lung, values))
        summary[name] = {key: float(np.nanmedian([item[key] for item in stats])) for key in stats[0]}

    # parte 2: dataset padrão
    generated_stats = {}
    for case in cases:
        readings, lung, values = analyse(case, case.synthetic)
        generated_stats[case.paths.case_id] = (case, readings, lung, values)
        rows.append({"source": "dataset padrão", "label": "gerada", "case_id": case.paths.case_id, **values})
    summary["Dataset padrão (mediana)"] = {key: float(np.nanmedian([item[3][key] for item in generated_stats.values()])) for key in rows[-1] if key.startswith(("area_", "correlacao_"))}
    typical = min(generated_stats.values(), key=lambda item: abs(np.nan_to_num(item[3]["correlacao_textura"], nan=9) - summary["Dataset padrão (mediana)"]["correlacao_textura"]))
    worst = min(generated_stats.values(), key=lambda item: np.nan_to_num(item[3]["correlacao_textura"], nan=9))
    real_entries = [
        (f"Gerada: corte típico ({short_name(typical[0].paths.case_id)})", typical[0], typical[0].synthetic, typical[1], typical[2], typical[3]),
        (f"Gerada: textura menos no lugar ({short_name(worst[0].paths.case_id)})", worst[0], worst[0].synthetic, worst[1], worst[2], worst[3]),
    ]

    # parte 3: exemplos claros do Dgen (sinalizados pela densidade)
    if args.examples_dir:
        flagged = []
        for case in load_cases(args.examples_dir, args.mask_dir, args.erosion_radius):
            status = density_status(density_features(case.synthetic, case.lung_mask, interval), density_features(case.original, case.lung_mask, interval), limits)
            if status in ("too_bright", "brighter"):
                readings, lung, values = analyse(case, case.synthetic)
                flagged.append((case, readings, lung, values))
                rows.append({"source": "exemplo", "label": status, "case_id": case.paths.case_id, **values})
        summary["Exemplos claros (mediana)"] = {key: float(np.nanmedian([item[3][key] for item in flagged])) for key in flagged[0][3]}
        for case, readings, lung, values in sorted(flagged, key=lambda item: -item[3]["area_mais_claro"])[:2]:
            real_entries.append((f"Exemplo claro ({short_name(case.paths.case_id)})", case, case.synthetic, readings, lung, values))

    figures = args.output_dir / "figuras"
    figures.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "areas_por_caso.csv", rows)
    (args.output_dir / "resumo.json").write_text(json.dumps({"block_px": args.block, "floor": floor, "medianas": summary}, indent=2, ensure_ascii=False), encoding="utf-8")
    save_figure(simulated_entries, interval, args.block, figures / "mapa_simulado.png", f"Mapa por região (blocos de {args.block} px): defeitos simulados")
    save_figure(real_entries, interval, args.block, figures / "mapa_reais.png", f"Mapa por região (blocos de {args.block} px): imagens geradas")
    for label, values in summary.items():
        print(f"{label:34s}", " ".join(f"{key.replace('area_', '')}={value:.2f}" for key, value in values.items()))


if __name__ == "__main__":
    main()
