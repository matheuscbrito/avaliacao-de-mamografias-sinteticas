"""Teste de stress de perturbação: compara perturbações controladas com aberrações reais do Dgen.

Aplica offsets de brilho e blur gaussiano em imagens reais para validar a
sensibilidade das métricas GLCM e de 1ª ordem ao fenômeno do pulmão esbranquiçado.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from .glcm_extractor import GLCMExtractor
from .load_data import CasePaths, discover_cases, load_image
from .validate_cases import RejectedCase, ValidCase, validate_case


def apply_perturbation(
    image: np.ndarray, mask: np.ndarray, offset: float = 0.0, blur_sigma: float = 0.0
) -> np.ndarray:
    """Aplica perturbações controladas (offset de brilho e/ou blur) apenas na ROI pulmonar."""
    perturbed = image.copy()
    if blur_sigma > 0:
        blurred = gaussian_filter(perturbed, sigma=blur_sigma)
        perturbed[mask] = blurred[mask]
    if offset != 0.0:
        perturbed[mask] += offset
    return perturbed


def quantize(image: np.ndarray, mask: np.ndarray, clip_low: float, clip_high: float, levels: int = 32) -> np.ndarray:
    """Quantiza a imagem dentro da máscara usando a régua fixa de Tukey."""
    clipped = np.clip(image, clip_low, clip_high)
    quantized = np.floor((clipped - clip_low) / (clip_high - clip_low) * (levels - 1)).astype(np.int32)
    return np.clip(quantized, 0, levels - 1)


def compute_stats(
    name: str, image: np.ndarray, mask: np.ndarray, extractor: GLCMExtractor
) -> dict[str, float]:
    """Calcula estatísticas de 1ª ordem e atributos GLCM para uma imagem."""
    pixels = image[mask]
    mu_roi = float(np.mean(pixels))
    sat_ceiling = float(np.mean(pixels >= extractor.clip_high) * 100.0) if extractor.clip_high is not None else 0.0

    features = extractor.extract(image, mask)
    result: dict[str, float] = {"name": name, "mu_ROI": mu_roi, "sat31_pct": sat_ceiling}
    result.update(features)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    """Escreve um CSV a partir de uma lista de dicionários."""
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def display_limits(image: np.ndarray) -> tuple[float, float]:
    """Percentis 1 e 99 da imagem para limites de visualização."""
    return tuple(float(v) for v in np.percentile(image, (1, 99)))


def save_main_panel(
    path: Path,
    images: list[tuple[np.ndarray, np.ndarray, str, dict]],
    vmin: float, vmax: float,
) -> None:
    """Gera o painel principal de comparação: perturbações progressivas vs aberrações reais."""
    n_row1 = 5  # Original + 4 offsets
    n_row2 = 4  # Original anômalo + Dgen + Dsr + Combo
    n_cols = max(n_row1, n_row2)

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    for ax in axes.flatten():
        ax.axis("off")

    for idx, (img, mask, title, stats) in enumerate(images):
        row = 0 if idx < n_row1 else 1
        col = idx if idx < n_row1 else idx - n_row1
        ax = axes[row, col]
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.contour(mask, levels=[0.5], colors=["#54A24B"], linewidths=0.8)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        mu = stats.get("mu_ROI", float("nan"))
        sat = stats.get("sat31_pct", float("nan"))
        ax.text(
            0.5, -0.12, f"μ={mu:.3f}\nSat31={sat:.1f}%",
            transform=ax.transAxes, ha="center", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    fig.suptitle("Perturbações Controladas vs Aberrações Reais do Dgen", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_histogram_panel(
    path: Path,
    entries: list[tuple[np.ndarray, np.ndarray, str, str]],
    clip_low: float, clip_high: float, levels: int = 32,
) -> None:
    """Gera o painel de histogramas de tons quantizados, destacando o acúmulo no Tom 31."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bins = np.arange(-0.5, levels + 0.5, 1)

    for img, mask, label, color in entries:
        quantized = quantize(img, mask, clip_low, clip_high, levels)
        roi_tones = quantized[mask]
        ax.hist(roi_tones, bins=bins, histtype="step", linewidth=2, color=color, density=True, label=label)

    ax.axvspan(levels - 1.5, levels - 0.5, color="#E45756", alpha=0.15, label="Zona de Saturação (Tom 31)")
    ax.set_xlabel("Tom Quantizado (0–31)")
    ax.set_ylabel("Densidade de Pixels")
    ax.set_title("Histograma de Tons Quantizados na ROI Pulmonar")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dgen-dir", type=Path, required=True, help="Pasta Dgen com TIFFs.")
    parser.add_argument("--dsr-dir", type=Path, required=True, help="Pasta Dsr com TIFFs.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Pasta de saída.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # --- Descobrir e validar casos ---
    candidates = discover_cases(args.dgen_dir)
    valid: list[ValidCase] = []
    for paths in candidates:
        result = validate_case(paths)
        if isinstance(result, ValidCase):
            valid.append(result)

    if not valid:
        raise RuntimeError("Nenhum caso válido encontrado.")
    print(f"Casos válidos: {len(valid)}")

    # --- Calibrar Tukey k=3.0 nos pixels reais ---
    real_lung_pixels = np.concatenate([case.original[case.lung_mask] for case in valid])
    extractor = GLCMExtractor(levels=32)
    extractor.calibrate_from_pixels(real_lung_pixels)
    clip_low, clip_high = extractor.clip_low, extractor.clip_high
    print(f"Régua Tukey k=3.0: [{clip_low:.4f}, {clip_high:.4f}]")

    # --- Calcular μ_ROI e Δμ para selecionar casos representativos ---
    case_info = []
    for case in valid:
        mask = case.lung_mask
        mu_orig = float(np.mean(case.original[mask]))
        mu_dgen = float(np.mean(case.synthetic[mask]))

        # Tentar carregar o Dsr correspondente
        dsr_path = args.dsr_dir / f"{case.paths.case_id}_generate.tiff"
        mu_dsr = None
        if dsr_path.exists():
            dsr_img = load_image(dsr_path)
            if dsr_img.shape == case.original.shape:
                mu_dsr = float(np.mean(dsr_img[mask]))

        case_info.append({
            "case": case, "mu_orig": mu_orig, "mu_dgen": mu_dgen,
            "delta_dgen": mu_dgen - mu_orig, "mu_dsr": mu_dsr,
            "dsr_path": dsr_path,
        })

    # Caso normal: mediana de μ_orig
    case_info.sort(key=lambda x: x["mu_orig"])
    normal_info = case_info[len(case_info) // 2]
    normal_case = normal_info["case"]

    # Caso anômalo: maior Δμ (Dgen)
    case_info.sort(key=lambda x: x["delta_dgen"], reverse=True)
    anomalous_info = case_info[0]
    anomalous_case = anomalous_info["case"]

    print(f"Caso normal:  {normal_case.paths.case_id[-30:]} (μ_orig={normal_info['mu_orig']:.3f})")
    print(f"Caso anômalo: {anomalous_case.paths.case_id[-30:]} (Δμ={anomalous_info['delta_dgen']:+.3f})")

    # --- Gerar perturbações no caso normal ---
    img_normal = normal_case.original
    mask_normal = normal_case.lung_mask

    offsets = [0.3, 0.7, 1.0, 1.4]
    blur_sigmas = [1.0, 2.0, 3.0]

    all_results: list[dict] = []

    # Caso normal original
    stats_normal = compute_stats("Original Normal", img_normal, mask_normal, extractor)
    all_results.append(stats_normal)

    # Perturbações de offset
    perturbed_images = {}
    for offset in offsets:
        name = f"Offset +{offset}"
        img = apply_perturbation(img_normal, mask_normal, offset=offset)
        perturbed_images[name] = img
        all_results.append(compute_stats(name, img, mask_normal, extractor))

    # Perturbações de blur
    for sigma in blur_sigmas:
        name = f"Blur σ={sigma}"
        img = apply_perturbation(img_normal, mask_normal, blur_sigma=sigma)
        perturbed_images[name] = img
        all_results.append(compute_stats(name, img, mask_normal, extractor))

    # Combinação: offset +1.0 e blur σ=2.0
    name_combo = "Combo +1.0 σ=2.0"
    img_combo = apply_perturbation(img_normal, mask_normal, offset=1.0, blur_sigma=2.0)
    perturbed_images[name_combo] = img_combo
    all_results.append(compute_stats(name_combo, img_combo, mask_normal, extractor))

    # --- Caso anômalo: Original, Dgen, Dsr ---
    mask_anomalous = anomalous_case.lung_mask
    img_anom_orig = anomalous_case.original
    img_anom_dgen = anomalous_case.synthetic

    stats_anom_orig = compute_stats("Anômalo Original", img_anom_orig, mask_anomalous, extractor)
    stats_anom_dgen = compute_stats("Anômalo Dgen (Real)", img_anom_dgen, mask_anomalous, extractor)
    all_results.append(stats_anom_orig)
    all_results.append(stats_anom_dgen)

    img_anom_dsr = None
    stats_anom_dsr = None
    if anomalous_info["dsr_path"].exists():
        img_anom_dsr = load_image(anomalous_info["dsr_path"])
        if img_anom_dsr.shape == img_anom_orig.shape:
            stats_anom_dsr = compute_stats("Anômalo Dsr", img_anom_dsr, mask_anomalous, extractor)
            all_results.append(stats_anom_dsr)

    # --- Salvar CSV com todas as features ---
    write_csv(args.output_dir / "perturbation_glcm_features.csv", all_results)
    print(f"CSV salvo: {args.output_dir / 'perturbation_glcm_features.csv'}")

    # --- Painel principal de comparação ---
    vmin, vmax = display_limits(img_normal)
    panel_images: list[tuple[np.ndarray, np.ndarray, str, dict]] = [
        (img_normal, mask_normal, "Original Normal", stats_normal),
        (perturbed_images["Offset +0.3"], mask_normal, "Offset +0.3", all_results[1]),
        (perturbed_images["Offset +0.7"], mask_normal, "Offset +0.7", all_results[2]),
        (perturbed_images["Offset +1.0"], mask_normal, "Offset +1.0", all_results[3]),
        (perturbed_images["Offset +1.4"], mask_normal, "Offset +1.4", all_results[4]),
        # Linha 2
        (img_anom_orig, mask_anomalous, "Anômalo Original", stats_anom_orig),
        (img_anom_dgen, mask_anomalous, "Dgen Aberrante", stats_anom_dgen),
    ]
    if img_anom_dsr is not None and stats_anom_dsr is not None:
        panel_images.append((img_anom_dsr, mask_anomalous, "Dsr Corrigido", stats_anom_dsr))
    panel_images.append((img_combo, mask_normal, "Combo +1.0 σ=2.0", all_results[8]))

    save_main_panel(figures_dir / "main_comparison_panel.png", panel_images, vmin, vmax)
    print(f"Painel principal salvo: {figures_dir / 'main_comparison_panel.png'}")

    # --- Painel de histogramas de tons quantizados ---
    hist_entries = [
        (img_normal, mask_normal, "Original Normal", "#4C78A8"),
        (perturbed_images["Offset +1.4"], mask_normal, "Perturbado +1.4", "#F58518"),
        (img_anom_dgen, mask_anomalous, "Dgen Aberrante", "#E45756"),
    ]
    if img_anom_dsr is not None:
        hist_entries.append((img_anom_dsr, mask_anomalous, "Dsr Corrigido", "#54A24B"))

    save_histogram_panel(figures_dir / "tone_histogram_panel.png", hist_entries, clip_low, clip_high)
    print(f"Histograma de tons salvo: {figures_dir / 'tone_histogram_panel.png'}")

    # --- Resumo no stdout ---
    print("\n" + "=" * 70)
    print("RESUMO: Sensibilidade das Métricas às Perturbações")
    print("=" * 70)
    header = f"{'Condição':<25} {'μ_ROI':>8} {'Sat31%':>8} {'Contraste':>10} {'Energia':>10} {'Gray Mean':>10}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        name = r.get("name", "?")
        mu = r.get("mu_ROI", float("nan"))
        sat = r.get("sat31_pct", float("nan"))
        contrast = r.get("glcm_contrast", float("nan"))
        energy = r.get("glcm_energy", float("nan"))
        gray_mean = r.get("glcm_gray_mean", float("nan"))
        print(f"{name:<25} {mu:>8.3f} {sat:>8.1f} {contrast:>10.3f} {energy:>10.4f} {gray_mean:>10.2f}")

    print(f"\nConcluído. Resultados em: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
