"""O que cada técnica enxerga: figuras de um corte na original e em cada versão do histórico.

Cada figura tem uma coluna para a original e uma para cada versão, todas na
mesma escala, recortadas no pulmão. Os títulos trazem o número que a técnica
dá para aquele corte (o mesmo da tabela do placar).

Mapa por região, em dois modos:
- **valor de cada imagem**: o mapa de cada imagem sozinha (brilho, textura fina
  ou grossa por bloco de 16 px). Vale para qualquer versão, inclusive só máscara;
- **comparado com a original**: cada bloco dividido pelo mesmo bloco da
  original. Só vale no modo mesma anatomia: numa versão só máscara o bloco (x, y)
  da sintética é outro tecido, e a divisão não mede erro do gerador.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pywt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from scipy.ndimage import uniform_filter

from wavelet.src.mapa_regioes import VERDICT_COLORS, VERDICTS, block_maps, compare_maps
from frameworks.texture_evaluator.glcm.src import quantize
from frameworks.texture_evaluator.placar import load_case, load_history
from frameworks.texture_evaluator.regional import BLOCK, MASK_ONLY, MIN_TISSUE
from frameworks.texture_evaluator.run_mvp import context_from_dict

TECHNIQUE_FIGURES = {
    "delta_mu": "Imagem na mesma janela de exibição; o título traz a média do pulmão e o Δμ.",
    "densidade": "Vermelho: pixels acima do P99 do pulmão real; azul: abaixo do P1. É o que a densidade conta.",
    "ll": "Banda LL da wavelet: a imagem vista em média local de 4×4 px, ou seja, só o brilho, sem a textura.",
    "wavelet_fina": "Energia do nível mais fino da wavelet (~2 px), pixel a pixel, dentro da máscara do pulmão e sobre a CT inteira (escuro = liso, amarelo = muito detalhe). Mesma escala em todas as colunas: borrar apaga os pontos claros.",
    "wavelet_grossa": "Energia do nível grosso da wavelet (~4 px), pixel a pixel, dentro da máscara do pulmão e sobre a CT inteira (escuro = liso, amarelo = muito detalhe). Mesma escala em todas as colunas.",
    "glcm_contraste": "Diferença de tom entre cada pixel e seus vizinhos, na régua de 32 tons da GLCM (média local de 5×5 px).",
}
MAP_LAYERS = {
    "brilho": "Brilho (média do bloco)",
    "fina": "Textura fina (Wavelet 2 px)",
    "grossa": "Textura grossa (Wavelet 4 px)",
    "veredito": "Veredito de textura",
}
MASK_ONLY_NOTE = "só máscara:\noutro pulmão no mesmo\ncontorno; não há o mesmo\nbloco da original para comparar"
# brilho comparado em preto e branco (escureceu = preto, clareou = branco), para não se confundir com as cores da textura
BRIGHT_STYLE = {"escureceu": dict(facecolor="black", edgecolor="black"), "clareou": dict(facecolor="white", edgecolor="black")}


def load(history_dir: Path, case_id: str):
    """Calibração, versões e o corte carregado em cada versão."""
    history = load_history(history_dir)
    versions = history["versions"]
    return context_from_dict(history["calibration"]), versions, [load_case(entry, case_id) for entry in versions]


def _grid(n_columns: int, height: float = 3.6):
    fig, axes = plt.subplots(1, n_columns, figsize=(3.3 * n_columns, height), squeeze=False)
    axes = axes[0]
    for axis in axes:
        axis.axis("off")
    return fig, axes


def _finish(fig, axes, mask, image=None, label=None):
    rows, cols = np.where(mask)
    for axis in axes:
        if axis.images:
            axis.contour(mask, levels=[.5], colors="#00a36c", linewidths=.5)
            axis.set_xlim(cols.min() - 6, cols.max() + 6)
            axis.set_ylim(rows.max() + 6, rows.min() - 6)
    if image is not None:
        fig.colorbar(image, ax=list(axes), shrink=.8, pad=.01, label=label)
    return fig


def _inside(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, array, np.nan)


def _upsample(band: np.ndarray, factor: int, shape) -> np.ndarray:
    return np.kron(band, np.ones((factor, factor)))[: shape[0], : shape[1]]


def _detail(image: np.ndarray, ruler: dict, level: int) -> np.ndarray:
    """Energia do detalhe wavelet em cada ponto (média de LH², HL², HH²): nível 1 = ~2 px (fino), nível 2 = ~4 px (grosso).

    É a mesma energia que entra no score, antes de ser resumida no pulmão.
    """
    normalised = (np.clip(image, ruler["p1"], ruler["p99"]) - ruler["p1"]) / (ruler["p99"] - ruler["p1"])
    coeffs = pywt.wavedec2(normalised, "haar", level=2, mode="periodization")
    bands = coeffs[2] if level == 1 else coeffs[1]
    return _upsample(np.mean([band**2 for band in bands], axis=0), 2**level, image.shape)


def _ll(image: np.ndarray, ruler: dict) -> np.ndarray:
    normalised = (np.clip(image, ruler["p1"], ruler["p99"]) - ruler["p1"]) / (ruler["p99"] - ruler["p1"])
    return _upsample(pywt.wavedec2(normalised, "haar", level=2, mode="periodization")[0] / 4, 4, image.shape)


def _local_contrast(image: np.ndarray, mask: np.ndarray, scale) -> np.ndarray:
    tones = quantize(image, scale).astype(float)
    total, count = np.zeros_like(tones), np.zeros_like(tones)
    for dy, dx in ((0, 1), (-1, 1), (-1, 0), (-1, -1)):
        shifted = np.roll(tones, (dy, dx), axis=(0, 1))
        valid = mask & np.roll(mask, (dy, dx), axis=(0, 1))
        total += np.where(valid, (tones - shifted) ** 2, 0)
        count += valid
    return uniform_filter(total, 5) / np.maximum(uniform_filter(count, 5), 1e-9)


def technique_figure(history_dir: Path, case_id: str, technique: str, rows: dict[str, dict]):
    """O que a técnica enxerga no corte, na original e em cada versão; `rows` são as linhas do placar por versão."""
    ctx, versions, cases = load(history_dir, case_id)
    original, mask = cases[0].original, cases[0].lung_mask
    images = [original] + [case.synthetic for case in cases]
    names = ["Original"] + [entry["name"] for entry in versions]
    fig, axes = _grid(len(images))
    window = dict(cmap="gray", vmin=ctx.interval["p1"], vmax=ctx.interval["p99"])
    shown, label = None, None

    if technique == "delta_mu":
        base = original[mask].mean()
        for axis, name, image in zip(axes, names, images):
            axis.imshow(image, **window)
            mean = image[mask].mean()
            axis.set_title(f"{name}\nmédia {mean:.3f}" + ("" if name == "Original" else f" · Δμ {mean - base:+.3f}"), fontsize=9)
    elif technique in ("wavelet_fina", "wavelet_grossa"):
        # como em wavelet/src/visualize_wavelet_example.py: CT inteira em cinza e a energia por pixel em cima, só no pulmão
        level = 1 if technique == "wavelet_fina" else 2
        energies = [_inside(_detail(image, ctx.ruler, level), mask) for image in images]
        high = np.nanpercentile(energies[0], 99)
        column = "wavelet_fine_pct" if level == 1 else "wavelet_coarse_pct"
        body = dict(cmap="gray", vmin=np.percentile(original, 1), vmax=np.percentile(original, 99))
        for axis, name, image, energy in zip(axes, names, images, energies):
            axis.imshow(image, **body)
            shown = axis.imshow(energy, cmap="inferno", vmin=0, vmax=high, alpha=.85)
            axis.set_title(name if name == "Original" else f"{name}\n{float(rows[name][column]):.0f}% da original", fontsize=9)
        label = f"energia do detalhe a ~{2 ** level} px"
        # imagem inteira, sem recorte no pulmão, para ver o mapa dentro da anatomia
        fig.colorbar(shown, ax=list(axes), shrink=.8, pad=.01, label=label)
        return fig
    elif technique == "densidade":
        for axis, name, image in zip(axes, names, images):
            axis.imshow(image, **window)
            above, below = mask & (image > ctx.interval["p99"]), mask & (image < ctx.interval["p1"])
            overlay = np.full(image.shape + (4,), 0.0)
            overlay[above] = (.84, .15, .16, 1)
            overlay[below] = (.12, .47, .71, 1)
            axis.imshow(overlay)
            axis.set_title(f"{name}\n{above.sum() / mask.sum():.1%} acima · {below.sum() / mask.sum():.1%} abaixo", fontsize=9)
        axes[-1].legend(handles=[Patch(color="#d62728", label="acima do P99 real"), Patch(color="#1f77b4", label="abaixo do P1 real")], loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, frameon=False)
    else:
        maps = {
            "ll": lambda image: _ll(image, ctx.ruler),
            "glcm_contraste": lambda image: _local_contrast(image, mask, ctx.scale),
        }[technique]
        values = [_inside(maps(image), mask) for image in images]
        low, high = (np.nanpercentile(values[0], 1), np.nanpercentile(values[0], 99)) if technique == "ll" else (0, np.nanpercentile(values[0], 99))
        column = {"ll": "abs_delta_ll", "glcm_contraste": "glcm_contrast_pct"}[technique]
        label = {"ll": "brilho local", "glcm_contraste": "diferença² de tom entre vizinhos"}[technique]
        for axis, name, value in zip(axes, names, values):
            shown = axis.imshow(value, cmap="gray", vmin=low, vmax=high)
            if name == "Original":
                axis.set_title(name, fontsize=9)
            elif technique == "ll":
                axis.set_title(f"{name}\n|ΔLL| {float(rows[name][column]):.3f}", fontsize=9)
            else:
                axis.set_title(f"{name}\n{float(rows[name][column]):.0f}% da original", fontsize=9)
    return _finish(fig, axes, mask, shown, label)


def map_figure(history_dir: Path, case_id: str, layer: str, compared: bool):
    """Mapa por região de um corte: cada imagem sozinha ou cada bloco comparado com o mesmo bloco da original."""
    ctx, versions, cases = load(history_dir, case_id)
    original, mask = cases[0].original, cases[0].lung_mask
    base, lung = block_maps(original, mask, ctx.ruler, BLOCK, "haar", MIN_TISSUE)
    extent = (0, lung.shape[1] * BLOCK, lung.shape[0] * BLOCK, 0)
    evaluated = np.kron(lung, np.ones((BLOCK, BLOCK)))
    window = dict(cmap="gray", vmin=ctx.interval["p1"], vmax=ctx.interval["p99"])
    fig, axes = _grid(len(cases) + 1, 3.9)
    shown, label = None, None

    if not compared:
        key = {"brilho": "ll", "fina": "fine", "grossa": "coarse"}[layer]
        maps = [base] + [block_maps(case.synthetic, mask, ctx.ruler, BLOCK, "haar", MIN_TISSUE)[0] for case in cases]
        low, high = np.percentile(base[key][lung], (2, 98)) if key == "ll" else (0, np.percentile(base[key][lung], 98))
        label = {"ll": "brilho médio do bloco", "fine": "detalhe a ~2 px (energia)", "coarse": "detalhe a ~4 px (energia)"}[key]
        for axis, name, values, image in zip(axes, ["Original"] + [entry["name"] for entry in versions], maps, [original] + [case.synthetic for case in cases]):
            axis.imshow(image, **window, alpha=.35)
            shown = axis.imshow(np.where(lung, values[key], np.nan), cmap="gray", vmin=low, vmax=high, extent=extent, interpolation="nearest")
            axis.set_title(name, fontsize=9)
        return _finish(fig, axes, mask, shown, label)

    axes[0].imshow(original, **window)
    axes[0].set_title("Original (referência)", fontsize=9)
    floor = ctx.maps["floor"]
    for axis, entry, case in zip(axes[1:], versions, cases):
        if entry.get("mode") == MASK_ONLY:
            axis.imshow(case.synthetic, **window, alpha=.25)
            axis.text(.5, .5, MASK_ONLY_NOTE, ha="center", va="center", fontsize=8, transform=axis.transAxes)
            axis.set_title(entry["name"], fontsize=9)
            continue
        generated, _ = block_maps(case.synthetic, mask, ctx.ruler, BLOCK, "haar", MIN_TISSUE)
        readings = compare_maps(base, generated, lung, floor)
        axis.imshow(case.synthetic, **window, alpha=.45)
        axis.contour(evaluated, levels=[.5], colors="black", linewidths=.6, linestyles=":")
        if layer == "brilho":
            shift = ctx.maps["bright_shift"]
            for y, x in zip(*np.where(lung)):
                name = "clareou" if readings["ll"][y, x] > shift else "escureceu" if readings["ll"][y, x] < -shift else None
                if name:
                    axis.add_patch(Rectangle((x * BLOCK, y * BLOCK), BLOCK, BLOCK, linewidth=.8, **BRIGHT_STYLE[name]))
            darker, brighter = (np.mean(readings["ll"][lung] < -shift), np.mean(readings["ll"][lung] > shift))
            axis.set_title(f"{entry['name']}\n{brighter:.0%} clareou · {darker:.0%} escureceu", fontsize=9)
        elif layer == "veredito":
            axis.imshow(np.where(lung, readings["verdict"], np.nan).astype(float), cmap=ListedColormap(VERDICT_COLORS), vmin=-.5, vmax=len(VERDICTS) - .5, extent=extent, alpha=.85, interpolation="nearest")
            axis.set_title(f"{entry['name']}\n{1 - np.mean(readings['verdict'][lung] == 0):.0%} da área com textura alterada", fontsize=9)
        else:
            key = "fine" if layer == "fina" else "coarse"
            shown = axis.imshow(np.where(lung, readings[key], np.nan), cmap="RdBu_r", vmin=0, vmax=200, extent=extent, alpha=.9, interpolation="nearest")
            label = "% do mesmo bloco da original\nazul: perdeu · vermelho: ganhou"
            axis.set_title(f"{entry['name']}\nmediana {np.median(readings[key][lung]):.0f}% da original", fontsize=9)
    if layer == "brilho":
        axes[-1].legend(handles=[Patch(label=name, linewidth=.8, **style) for name, style in BRIGHT_STYLE.items()] + [Patch(facecolor="none", edgecolor="black", linestyle=":", label="área avaliada")], loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, frameon=False)
    elif layer == "veredito":
        axes[-1].legend(handles=[Patch(color=color, label=name) for name, color in zip(VERDICTS, VERDICT_COLORS)], loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, frameon=False)
    return _finish(fig, axes, mask, shown, label)
