"""EDA reproduzível e explicada dos pares de CT pulmonar real/sintético.

Uso:
    .venv/bin/python lung_eda.py

O que este script faz, em linguagem simples
--------------------------------------------
Para cada corte de tomografia, há três arquivos: a imagem real (referência),
a imagem sintética (gerada pelo modelo) e uma máscara fornecida junto ao
dataset. O script confirma que os três existem, compara real e sintética pixel
a pixel e gera gráficos para revisão visual.

Ele responde: "quanto a imagem sintética se parece com a real?". Ainda NÃO
responde: "a textura pulmonar é clinicamente equivalente?". Essa segunda
pergunta exige uma máscara de pulmão validada, seleção de cortes pulmonares e
extração radiômica, que serão a próxima etapa.

Os resultados ficam em output/lung_eda/:
  - pair_metrics.csv: uma linha por par, com todas as métricas;
  - summary.txt: resumo numérico curto;
  - README.md: explicação de cada métrica e como interpretar;
  - pair_distributions.png: histogramas das métricas;
  - paired_examples.png: exemplos visuais do pior, mediano e melhor par.

O script só lê o dataset e escreve dentro de output/lung_eda/; nunca altera os
TIFFs ou PNGs de entrada.
"""
from __future__ import annotations

import csv
from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


DATASET = Path("upscale_test_new_lung_Dgen")
OUTPUT = Path("output/lung_eda")


README = """\
# EDA do dataset de CT pulmonar

Este diretório é gerado por `lung_eda.py`. A análise compara cada imagem TIFF
real com sua correspondente sintética. O objetivo é uma triagem inicial da
fidelidade das imagens, antes da validação radiômica de textura.

## Como ler as métricas

| Campo | Em palavras simples | Como interpretar |
| --- | --- | --- |
| `psnr_full` | Mede a força do sinal da imagem em relação ao erro entre real e sintética; é expresso em decibéis (dB). | **Maior é melhor**. Um valor alto indica que, em média, os pixels são mais parecidos. Não garante que a anatomia ou a textura sejam corretas. |
| `ssim_full` | Mede quão semelhante é a estrutura visual: bordas, contraste local e padrões. Vai aproximadamente de 0 a 1. | **Mais perto de 1 é melhor**. É mais alinhado à aparência visual que PSNR, mas ainda não é uma métrica clínica. |
| `mae_full` | Erro absoluto médio: em cada pixel, calcula a diferença entre real e sintética e tira a média. | **Menor é melhor**. Está na mesma escala numérica do TIFF, então só pode ser comparado entre conjuntos pré-processados da mesma forma. |
| `rmse_full` | Parecido com MAE, mas dá peso maior aos erros muito grandes. | **Menor é melhor**. Se RMSE for muito maior que MAE, alguns poucos pontos têm erro especialmente alto. |
| `mask_nonzero_area_fraction` | Fração da imagem onde a máscara fornecida não é zero. | **Não é qualidade.** Serve apenas para entender quanto da imagem foi marcada. A máscara atual pode incluir pulmão, tórax e mesa. |
| `mae_mask_nonzero` | MAE calculado apenas dentro da parte não-zero da máscara fornecida. | **Menor é melhor**, mas ainda não é MAE pulmonar: só será uma medida de pulmão após validar ou gerar uma máscara pulmonar confiável. |
| `original_mean_mask_nonzero` / `generated_mean_mask_nonzero` | Intensidade média na região marcada, na imagem real e sintética. | Ajuda a detectar mudança global de brilho/escala dentro da região marcada. |
| `original_std_mask_nonzero` / `generated_std_mask_nonzero` | Variação das intensidades na região marcada. | Uma queda forte na sintética pode indicar suavização; aumento pode indicar ruído/heterogeneidade. É um indício, não uma conclusão de textura. |
| `mask_labels` | Valores que aparecem na máscara daquele caso. | Registrar isto é importante porque o dataset usa valores como 0, 0,5 e 1; o significado anatômico ainda precisa ser confirmado. |

## Limites desta EDA

As imagens são TIFFs float, sem metadados DICOM nem unidades Hounsfield (HU).
Por isso, os números descrevem fidelidade na escala disponibilizada pelo
dataset, e não uma validação física de CT. Os PNGs são apenas prévias visuais;
as métricas usam os TIFFs.

## Próxima etapa recomendada

1. Confirmar com quem gerou os dados o significado dos valores 0,5 e 1 da
   máscara fornecida.
2. Gerar/revisar uma máscara pulmonar com `lungmask` em uma amostra.
3. Manter apenas cortes que realmente contenham pulmão e erodir levemente sua
   borda.
4. Extrair GLCM, GLRLM, GLSZM e NGTDM dentro dessa ROI pulmonar, usando os
   mesmos parâmetros para cada par real/sintético.
"""


def read(path: Path) -> np.ndarray:
    """Abre um TIFF e o converte para números float, preservando sua escala."""
    return np.asarray(Image.open(path), dtype=np.float64)


def percentile_range(image: np.ndarray) -> tuple[float, float]:
    """Escolhe uma faixa de exibição que ignora pixels extremamente atípicos.

    Isto afeta somente a aparência dos painéis PNG; nunca altera os dados nem
    as métricas calculadas.
    """
    lo, hi = np.percentile(image, (1, 99))
    return float(lo), float(hi if hi > lo else lo + 1)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Mantemos esta explicação perto dos resultados para qualquer pessoa poder
    # interpretar o CSV sem precisar abrir o código.
    (OUTPUT / "README.md").write_text(dedent(README), encoding="utf-8")
    rows: list[dict[str, float | str]] = []
    examples: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    missing: list[str] = []

    originals = sorted(
        p for p in DATASET.glob("*.tiff")
        if not p.name.endswith(("_generate.tiff", "_mask.tiff"))
    )
    for orig_path in originals:
        key = orig_path.stem
        gen_path = orig_path.with_name(f"{key}_generate.tiff")
        mask_path = orig_path.with_name(f"{key}_mask.tiff")
        if not gen_path.exists() or not mask_path.exists():
            missing.append(key)
            continue
        original, generated, mask = read(orig_path), read(gen_path), read(mask_path)
        mask_nonzero = mask > 0
        # PSNR e SSIM precisam saber a faixa de intensidade considerada. Usar
        # a faixa do TIFF real torna a comparação coerente dentro de cada par.
        data_range = float(original.max() - original.min())
        abs_diff = np.abs(original - generated)
        row: dict[str, float | str] = {
            "case_id": key,
            # PSNR (dB): compara a intensidade do sinal com o erro. Quanto
            # maior, menor o erro relativo médio entre os pixels.
            "psnr_full": float(peak_signal_noise_ratio(original, generated, data_range=data_range)),
            # SSIM (~0 a 1): compara estrutura visual, como bordas e contraste
            # local. Valor 1 representa imagens estruturalmente idênticas.
            "ssim_full": float(structural_similarity(original, generated, data_range=data_range)),
            # MAE: diferença absoluta média. Fácil de interpretar, mas depende
            # da escala de intensidade dos TIFFs.
            "mae_full": float(abs_diff.mean()),
            # RMSE: também mede erro, mas penaliza mais os pixels com grandes
            # diferenças; útil para revelar erros localizados intensos.
            "rmse_full": float(np.sqrt(np.mean((original - generated) ** 2))),
            # Não assumimos que esta máscara seja somente pulmão: a inspeção
            # visual revelou que ela também pode cobrir tórax/mesa.
            "mask_nonzero_area_fraction": float(mask_nonzero.mean()),
            "original_mean_mask_nonzero": float(original[mask_nonzero].mean()),
            "generated_mean_mask_nonzero": float(generated[mask_nonzero].mean()),
            "original_std_mask_nonzero": float(original[mask_nonzero].std()),
            "generated_std_mask_nonzero": float(generated[mask_nonzero].std()),
            "mae_mask_nonzero": float(abs_diff[mask_nonzero].mean()),
            "mask_labels": "|".join(map(str, np.unique(mask).tolist())),
        }
        rows.append(row)
        examples.append((key, original, generated, mask))

    if not rows:
        raise RuntimeError("Nenhum par TIFF completo foi encontrado.")

    columns = list(rows[0])
    with (OUTPUT / "pair_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    def values(name: str) -> np.ndarray:
        """Reúne uma coluna do CSV como vetor numérico para resumos/gráficos."""
        return np.array([float(row[name]) for row in rows])

    summary = {
        "n_complete_pairs": len(rows),
        "n_missing_companion_files": len(missing),
        "psnr_full_mean": values("psnr_full").mean(),
        "psnr_full_median": np.median(values("psnr_full")),
        "ssim_full_mean": values("ssim_full").mean(),
        "ssim_full_median": np.median(values("ssim_full")),
        "mae_full_mean": values("mae_full").mean(),
        "mae_mask_nonzero_mean": values("mae_mask_nonzero").mean(),
        "mask_nonzero_area_fraction_median": np.median(values("mask_nonzero_area_fraction")),
    }
    with (OUTPUT / "summary.txt").open("w") as f:
        f.write("Lung CT paired-dataset EDA\n")
        for name, value in summary.items():
            f.write(f"{name}: {value}\n")
        if missing:
            f.write("missing_cases:\n" + "\n".join(missing) + "\n")

    # Distribuições que avaliam fidelidade ponto a ponto e cobertura da máscara.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, column, title, xlabel in [
        (axes[0], "psnr_full", "PSNR por par", "PSNR (dB)"),
        (axes[1], "ssim_full", "SSIM por par", "SSIM"),
        (axes[2], "mask_nonzero_area_fraction", "Cobertura da máscara", "Fração da imagem"),
    ]:
        ax.hist(values(column), bins=14, color="#4C78A8", edgecolor="white")
        ax.axvline(np.median(values(column)), color="#E45756", label="mediana")
        ax.set(title=title, xlabel=xlabel, ylabel="Número de pares")
        ax.legend(frameon=False)
    fig.suptitle("EDA inicial — 100 pares CT real/sintético")
    fig.tight_layout()
    fig.savefig(OUTPUT / "pair_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Casos representativos: melhor, mediano e pior PSNR para revisão visual.
    ranked = sorted(zip(rows, examples), key=lambda item: float(item[0]["psnr_full"]))
    chosen = [ranked[0][1], ranked[len(ranked) // 2][1], ranked[-1][1]]
    fig, axes = plt.subplots(len(chosen), 4, figsize=(13, 9))
    for i, (case_id, original, generated, mask) in enumerate(chosen):
        lo, hi = percentile_range(original)
        panels = [
            (original, "Original", "gray", lo, hi),
            (generated, "Sintética", "gray", lo, hi),
            (mask, "Máscara fornecida", "viridis", None, None),
            (np.abs(original - generated), "|Original − sintética|", "magma", 0, np.percentile(np.abs(original-generated), 99)),
        ]
        for ax, (array, title, cmap, vmin, vmax) in zip(axes[i], panels):
            ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
        axes[i, 0].set_ylabel(case_id[-22:], fontsize=7)
    fig.suptitle("Inspeção visual: pior, mediano e melhor PSNR (de cima para baixo)")
    fig.tight_layout()
    fig.savefig(OUTPUT / "paired_examples.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"EDA concluído: {len(rows)} pares completos em {OUTPUT}")


if __name__ == "__main__":
    main()
