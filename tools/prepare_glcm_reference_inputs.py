"""Prepara os cortes pulmonares originais para a referência GLCM.

Cria `datasets/ct/glcm_reference_inputs/` com:
  - images/: links para os TIFFs originais selecionados (sem duplicar dados);
  - masks/: máscaras binárias, com 1 apenas no pulmão;
  - manifest.csv: critério e proveniência de cada caso;
  - qc_overview.png: revisão visual da seleção.

Critério provisório e verificável de seleção:
  - máscara fornecida com rótulo 1;
  - ao menos dois componentes pulmonares com >= 800 pixels;
  - área pulmonar total entre 3% e 35% da imagem.

O resultado são cortes torácicos centrais com ambos os pulmões visíveis. A
regra é deliberadamente conservadora: exclui ápices/bases e cortes abdominais,
pois a primeira referência de textura deve comparar regiões pulmonares estáveis.
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops


SOURCE = Path("datasets/ct/Dgen")
DESTINATION = Path("datasets/ct/glcm_reference_inputs")
MASK_LABEL = 1.0
MIN_COMPONENT_PIXELS = 800
MIN_AREA_FRACTION = 0.03
MAX_AREA_FRACTION = 0.35


def originals() -> list[Path]:
    return sorted(
        path for path in SOURCE.glob("*.tiff")
        if not path.name.endswith(("_generate.tiff", "_mask.tiff"))
    )


def select(image_path: Path) -> tuple[np.ndarray, float, int] | None:
    """Retorna a máscara binária se o corte contém os dois pulmões claramente."""
    raw_mask = np.asarray(Image.open(image_path.with_name(f"{image_path.stem}_mask.tiff")))
    lung = np.isclose(raw_mask, MASK_LABEL)
    components = [region for region in regionprops(label(lung)) if region.area >= MIN_COMPONENT_PIXELS]
    area_fraction = float(lung.mean())
    if len(components) < 2 or not MIN_AREA_FRACTION <= area_fraction <= MAX_AREA_FRACTION:
        return None
    return lung, area_fraction, len(components)


def main() -> None:
    if DESTINATION.exists():
        raise FileExistsError(f"{DESTINATION} já existe; não vou sobrescrever uma seleção existente.")
    image_dir = DESTINATION / "images"
    mask_dir = DESTINATION / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir()
    selected: list[tuple[Path, np.ndarray, float, int]] = []

    for image_path in originals():
        result = select(image_path)
        if result is None:
            continue
        lung, area_fraction, components = result
        selected.append((image_path, lung, area_fraction, components))
        # Link simbólico evita uma segunda cópia dos TIFFs; o GLCM os lê como
        # arquivos normais e o dataset de origem continua sendo a fonte única.
        (image_dir / image_path.name).symlink_to(Path("../..") / image_path)
        # Gravamos literalmente 0/1: o script GLCM usa o valor 1 como rótulo
        # de pulmão; 255 seria apenas uma convenção de exibição, não um label.
        binary_mask = lung.astype(np.uint8)
        Image.fromarray(binary_mask, mode="L").save(mask_dir / f"{image_path.stem}_lungmask.tiff")

    with (DESTINATION / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("case_id", "source_image", "mask_source", "lung_area_fraction", "n_components", "selection_rule"))
        writer.writeheader()
        for image_path, _, area_fraction, components in selected:
            writer.writerow({
                "case_id": image_path.stem,
                "source_image": str(image_path),
                "mask_source": f"{image_path.stem}_mask.tiff; label={MASK_LABEL}",
                "lung_area_fraction": f"{area_fraction:.6f}",
                "n_components": components,
                "selection_rule": "label 1; >=2 components of >=800px; area 3%-35%",
            })

    fig, axes = plt.subplots(6, 7, figsize=(14, 12))
    for axis, (image_path, lung, area_fraction, _) in zip(axes.flat, selected):
        image = np.asarray(Image.open(image_path), dtype=float)
        low, high = np.percentile(image, (1, 99))
        axis.imshow(image, cmap="gray", vmin=low, vmax=high)
        axis.contour(lung, levels=[0.5], colors="lime", linewidths=0.6)
        axis.set_title(f"{image_path.stem[-12:]}\n{area_fraction:.0%}", fontsize=6)
        axis.axis("off")
    for axis in axes.flat[len(selected):]:
        axis.axis("off")
    fig.suptitle("Entradas pré-validadas para referência GLCM — originais + ROI pulmonar", fontsize=13)
    fig.tight_layout()
    fig.savefig(DESTINATION / "qc_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    (DESTINATION / "README.md").write_text(
        "# Entradas para a referência GLCM\n\n"
        f"Foram selecionados **{len(selected)}** cortes originais centrais com ambos os pulmões delineados pela máscara fornecida. "
        "As imagens em `images/` são links simbólicos para o dataset original; as máscaras em `masks/` são binárias e usam 1 para pulmão. "
        "Veja `qc_overview.png` antes de alterar a regra de seleção.\n",
        encoding="utf-8",
    )
    print(f"Preparação concluída: {len(selected)} cortes em {DESTINATION}")


if __name__ == "__main__":
    main()
