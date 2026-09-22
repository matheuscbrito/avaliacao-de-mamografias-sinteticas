"""Cria a referência de textura GLCM das imagens CT ORIGINAIS.

O script não lê imagens sintéticas. Ele mede contraste e energia do GLCM em
uma máscara pulmonar, cria a distribuição de referência do conjunto real,
marca possíveis outliers para revisão e pode gerar uma prévia visual de cada
outlier.

Uso padrão, com as entradas pré-validadas criadas por
`prepare_glcm_reference_inputs.py`:

    .venv/bin/python glcm_lung_reference.py

Para usar outro conjunto de imagens/máscaras, passe `--images`, `--mask-dir`,
`--mask-suffix` e `--mask-label` explicitamente.

Por que `--mask-label` é obrigatório?
    As máscaras atualmente fornecidas usam mais de uma classe (0, 0,5 e 1).
    O script não pode adivinhar se um valor representa pulmão, corpo ou mesa.
    Informe aqui somente o rótulo que foi validado como pulmão. Para máscaras
    binárias geradas por lungmask, o valor usual será 1.

O protocolo é o mesmo para todas as originais:
    1. mantém somente pixels da máscara pulmonar escolhida;
    2. remove a borda da máscara por erosão;
    3. usa uma única faixa de intensidades e 32 níveis de cinza para todo o
       conjunto (não normaliza imagem por imagem);
    4. mede GLCM nas direções 0, 45, 90 e 135 graus;
    5. registra a média por direção de contraste e energia.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.morphology import disk, erosion


ANGLES = (0, 45, 90, 135)
# (deslocamento em linhas, deslocamento em colunas), para distância 1 pixel.
OFFSETS = ((0, 1), (-1, 1), (-1, 0), (-1, -1))


@dataclass
class Case:
    """Uma imagem real e sua ROI pulmonar, antes da extração GLCM."""

    case_id: str
    image: np.ndarray
    mask: np.ndarray
    image_path: Path


def load_2d(path: Path) -> np.ndarray:
    """Abre TIFF/PNG como uma matriz 2D de números float."""
    image = np.asarray(Image.open(path), dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"esperava imagem 2D; recebeu {image.shape}")
    return image


def matching_mask(image_path: Path, mask_dir: Path, suffix: str) -> Path:
    """Localiza a máscara pelo mesmo nome-base da imagem original."""
    return mask_dir / f"{image_path.stem}{suffix}"


def crop_pair_for_offset(
    quantized: np.ndarray, mask: np.ndarray, row_offset: int, col_offset: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Alinha uma imagem e sua vizinha deslocada, sem preencher bordas falsas."""
    rows, cols = quantized.shape
    row_start_a, row_end_a = max(0, -row_offset), min(rows, rows - row_offset)
    col_start_a, col_end_a = max(0, -col_offset), min(cols, cols - col_offset)
    row_start_b, row_end_b = max(0, row_offset), min(rows, rows + row_offset)
    col_start_b, col_end_b = max(0, col_offset), min(cols, cols + col_offset)
    return (
        quantized[row_start_a:row_end_a, col_start_a:col_end_a],
        quantized[row_start_b:row_end_b, col_start_b:col_end_b],
        mask[row_start_a:row_end_a, col_start_a:col_end_a],
        mask[row_start_b:row_end_b, col_start_b:col_end_b],
    )


def glcm_contrast_energy(
    image: np.ndarray, mask: np.ndarray, low: float, high: float, levels: int
) -> dict[str, float]:
    """Calcula GLCM somente para pares cujos DOIS pixels estão na máscara.

    Contraste é alto quando vizinhos costumam ter tons bem diferentes.
    Energia é alta quando poucas combinações de tons dominam a textura.
    """
    clipped = np.clip(image, low, high)
    quantized = np.floor((clipped - low) / (high - low) * (levels - 1)).astype(np.uint8)
    contrast_by_angle: list[float] = []
    energy_by_angle: list[float] = []
    gray = np.arange(levels, dtype=np.float64)
    squared_difference = (gray[:, None] - gray[None, :]) ** 2

    for row_offset, col_offset in OFFSETS:
        a, b, mask_a, mask_b = crop_pair_for_offset(quantized, mask, row_offset, col_offset)
        valid = mask_a & mask_b
        if not valid.any():
            raise ValueError("a máscara não contém pares vizinhos suficientes")
        matrix = np.zeros((levels, levels), dtype=np.float64)
        np.add.at(matrix, (a[valid], b[valid]), 1)
        # GLCM simétrica: trata as relações i→j e j→i como equivalentes.
        matrix += matrix.T
        matrix /= matrix.sum()
        contrast_by_angle.append(float((matrix * squared_difference).sum()))
        energy_by_angle.append(float((matrix**2).sum()))

    output: dict[str, float] = {
        "contrast": float(np.mean(contrast_by_angle)),
        "energy": float(np.mean(energy_by_angle)),
        "contrast_anisotropy": float(np.std(contrast_by_angle) / (np.mean(contrast_by_angle) + 1e-12)),
    }
    for angle, contrast, energy in zip(ANGLES, contrast_by_angle, energy_by_angle):
        output[f"contrast_{angle}deg"] = contrast
        output[f"energy_{angle}deg"] = energy
    return output


def tukey_limits(values: np.ndarray, multiplier: float) -> tuple[float, float]:
    """Limites de revisão: Q1 − 1,5×IQR e Q3 + 1,5×IQR por padrão."""
    q1, q3 = np.percentile(values, (25, 75))
    iqr = q3 - q1
    return float(q1 - multiplier * iqr), float(q3 + multiplier * iqr)


def describe(values: np.ndarray) -> dict[str, float | int]:
    """Resumo robusto para a distribuição de referência das imagens reais."""
    q1, q3 = np.percentile(values, (25, 75))
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "median": float(np.median(values)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "p5": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def save_distribution_plot(rows: list[dict], output: Path, limits: dict[str, tuple[float, float]]) -> None:
    """Salva histogramas com média, mediana e limites de revisão Tukey."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for axis, field, title, label in (
        (axes[0], "contrast", "Distribuição de contraste GLCM", "Contraste GLCM"),
        (axes[1], "energy", "Distribuição de energia GLCM", "Energia GLCM"),
    ):
        values = np.array([row[field] for row in rows], dtype=float)
        axis.hist(values, bins="auto", color="#4C78A8", edgecolor="white")
        axis.axvline(values.mean(), color="#E45756", linewidth=2, label="média")
        axis.axvline(np.median(values), color="#54A24B", linewidth=2, label="mediana")
        lower, upper = limits[field]
        axis.axvline(lower, color="#B279A2", linestyle="--", label="limite Tukey")
        axis.axvline(upper, color="#B279A2", linestyle="--")
        axis.set(title=title, xlabel=label, ylabel="Número de imagens reais")
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Referência de textura — imagens CT originais")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_outlier_preview(case: Case, row: dict, output: Path) -> None:
    """Gera uma prévia para revisão humana; a imagem não é alterada."""
    low, high = np.percentile(case.image, (1, 99))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(case.image, cmap="gray", vmin=low, vmax=high)
    axes[0].contour(case.mask, levels=[0.5], colors="lime", linewidths=0.8)
    axes[0].set_title("Original + ROI pulmonar erodida")
    axes[0].axis("off")
    axes[1].imshow(case.mask, cmap="gray")
    axes[1].set_title("ROI usada no GLCM")
    axes[1].axis("off")
    reason = row["outlier_reason"] or "sem alerta"
    fig.suptitle(
        f"{case.case_id}\ncontraste={row['contrast']:.4f} | energia={row['energy']:.4f} | alerta: {reason}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, default=Path("datasets/ct/glcm_reference_inputs/images"), help="Pasta dos TIFFs originais preparados.")
    parser.add_argument("--mask-dir", type=Path, default=Path("datasets/ct/glcm_reference_inputs/masks"), help="Pasta das máscaras pulmonares verificadas.")
    parser.add_argument("--mask-suffix", default="_lungmask.tiff", help="Sufixo da máscara após o nome-base da original.")
    parser.add_argument("--mask-label", type=float, default=1, help="Valor que representa pulmão na máscara.")
    parser.add_argument("--output", type=Path, default=Path("reports/legacy/dgen_glcm_reference"), help="Pasta de resultados.")
    parser.add_argument("--levels", type=int, default=32, help="Níveis de cinza após discretização global.")
    parser.add_argument("--erosion-radius", type=int, default=2, help="Erosão da borda pulmonar, em pixels.")
    parser.add_argument("--clip-percentiles", nargs=2, type=float, default=(1, 99), metavar=("LOW", "HIGH"), help="Percentis globais das originais usados no clipping.")
    parser.add_argument("--tukey-k", type=float, default=1.5, help="Multiplicador IQR para sinalizar outliers.")
    parser.add_argument("--previews", type=Path, help="Se informado, salva uma prévia de cada outlier nesta pasta.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.levels < 2:
        raise ValueError("--levels deve ser pelo menos 2")
    if not (0 <= args.clip_percentiles[0] < args.clip_percentiles[1] <= 100):
        raise ValueError("percentis de clipping inválidos")
    args.output.mkdir(parents=True, exist_ok=True)

    originals = sorted(
        path for path in args.images.glob("*.tiff")
        if not path.name.endswith(("_generate.tiff", "_mask.tiff"))
    )
    if not originals:
        raise RuntimeError(f"nenhuma original TIFF encontrada em {args.images}")

    cases: list[Case] = []
    rejected: list[dict[str, str]] = []
    for image_path in originals:
        mask_path = matching_mask(image_path, args.mask_dir, args.mask_suffix)
        if not mask_path.exists():
            rejected.append({"case_id": image_path.stem, "reason": f"máscara ausente: {mask_path.name}"})
            continue
        try:
            image, raw_mask = load_2d(image_path), load_2d(mask_path)
            if image.shape != raw_mask.shape:
                raise ValueError(f"dimensões diferentes: imagem={image.shape}, máscara={raw_mask.shape}")
            mask = np.isclose(raw_mask, args.mask_label)
            if args.erosion_radius:
                mask = erosion(mask, footprint=disk(args.erosion_radius))
            if mask.sum() < 1000:
                raise ValueError("ROI pulmonar tem menos de 1000 pixels após erosão")
            cases.append(Case(image_path.stem, image, mask, image_path))
        except Exception as error:
            rejected.append({"case_id": image_path.stem, "reason": str(error)})

    if not cases:
        raise RuntimeError("nenhuma imagem possui máscara pulmonar válida com os parâmetros fornecidos")

    # A faixa vem da união das regiões reais, e será guardada no JSON para que
    # a análise das sintéticas use exatamente a mesma discretização depois.
    real_lung_pixels = np.concatenate([case.image[case.mask] for case in cases])
    low, high = np.percentile(real_lung_pixels, args.clip_percentiles)
    if high <= low:
        raise RuntimeError("a faixa global de intensidade é degenerada")

    rows: list[dict] = []
    by_id = {case.case_id: case for case in cases}
    for case in cases:
        metrics = glcm_contrast_energy(case.image, case.mask, float(low), float(high), args.levels)
        rows.append({"case_id": case.case_id, "roi_pixels": int(case.mask.sum()), **metrics})

    contrast_values = np.array([row["contrast"] for row in rows])
    energy_values = np.array([row["energy"] for row in rows])
    limits = {
        "contrast": tukey_limits(contrast_values, args.tukey_k),
        "energy": tukey_limits(energy_values, args.tukey_k),
    }
    for row in rows:
        reasons = []
        for field, label in (("contrast", "contrast"), ("energy", "energy")):
            lower, upper = limits[field]
            if row[field] < lower or row[field] > upper:
                reasons.append(label)
        row["outlier"] = bool(reasons)
        row["outlier_reason"] = "+".join(reasons)

    columns = list(rows[0])
    with (args.output / "glcm_per_image.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "rejected_images.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("case_id", "reason"))
        writer.writeheader()
        writer.writerows(rejected)

    report = {
        "purpose": "Referência GLCM construída somente com CTs originais.",
        "n_originals_found": len(originals),
        "n_valid": len(rows),
        "n_rejected": len(rejected),
        "n_outliers_flagged": int(sum(row["outlier"] for row in rows)),
        "protocol": {
            "mask_label": args.mask_label,
            "erosion_radius_px": args.erosion_radius,
            "levels": args.levels,
            "angles_deg": list(ANGLES),
            "global_clip_percentiles": list(args.clip_percentiles),
            "global_intensity_range": [float(low), float(high)],
            "tukey_k": args.tukey_k,
        },
        "reference_all_valid_originals": {
            "contrast": describe(contrast_values),
            "energy": describe(energy_values),
        },
        "outlier_limits_tukey": {field: list(limit) for field, limit in limits.items()},
        "outliers": [row for row in rows if row["outlier"]],
        "important_note": "Outlier é um alerta para revisão visual; esta versão não remove outliers da referência automaticamente.",
    }
    (args.output / "glcm_reference.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    save_distribution_plot(rows, args.output / "glcm_distributions.png", limits)

    if args.previews:
        args.previews.mkdir(parents=True, exist_ok=True)
        for row in rows:
            if row["outlier"]:
                save_outlier_preview(by_id[row["case_id"]], row, args.previews / f"{row['case_id']}.png")

    print("REFERÊNCIA GLCM — IMAGENS ORIGINAIS")
    print(f"originais encontradas: {len(originals)} | válidas: {len(rows)} | rejeitadas: {len(rejected)}")
    print(f"outliers sinalizados para revisão: {sum(row['outlier'] for row in rows)}")
    for name, values in (("Contraste", contrast_values), ("Energia", energy_values)):
        stats = describe(values)
        print(f"{name}: média={stats['mean']:.4f} | mediana={stats['median']:.4f} | IQR=[{stats['q1']:.4f}, {stats['q3']:.4f}]")
    print(f"Resultados: {args.output}")


if __name__ == "__main__":
    main()
