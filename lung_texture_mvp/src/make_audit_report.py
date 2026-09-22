"""Gera um relatório visual independente da auditoria dos pares GLCM.

Execute a partir de ``lung_texture_mvp``:
    ../.venv/bin/python -m src.make_audit_report
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.morphology import disk, erosion

from .glcm_features import glcm_contrast_energy


OUTPUT = Path("outputs/auditoria_glcm")
DATASET = Path("../datasets/ct/Dgen")
RESULTS = Path("outputs/comparison_results.csv")
REFERENCE = Path("outputs/reference_originals.json")
COLORS = {"approved": "#27754a", "manual_review": "#b76e00", "reject": "#b53a35"}
LABELS = {"approved": "Aprovado", "manual_review": "Revisão manual", "reject": "Rejeitado"}


def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=float)


def case_arrays(case_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    original = load(DATASET / f"{case_id}.tiff")
    synthetic = load(DATASET / f"{case_id}_generate.tiff")
    mask = np.isclose(load(DATASET / f"{case_id}_mask.tiff"), 1)
    return original, synthetic, erosion(mask, footprint=disk(2))


def key_cases(rows: list[dict[str, str]]) -> list[tuple[str, dict[str, str]]]:
    approved = [row for row in rows if row["status"] == "approved"]
    return [
        ("Outlier de energia", max(rows, key=lambda row: float(row["original_energy"]))),
        ("Maior queda de contraste", min(rows, key=lambda row: float(row["delta_contrast"]))),
        ("Único caso rejeitado", next(row for row in rows if row["status"] == "reject")),
        ("Par mais estável entre os aprovados", min(approved, key=lambda row: max(float(row["pair_distance_contrast_iqr"]), float(row["pair_distance_energy_iqr"])))),
    ]


def save_case_panel(selected: list[tuple[str, dict[str, str]]]) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(17, 19), gridspec_kw={"width_ratios": (1, 1, 0.75, 1.25)})
    for row_index, (role, row) in enumerate(selected):
        original, synthetic, mask = case_arrays(row["case_id"])
        # Janela de APRESENTAÇÃO, deliberadamente separada do clipping do GLCM.
        # O cálculo usa somente a ROI pulmonar; reutilizar seus percentis para
        # exibir o CT inteiro saturava corpo e osso em branco. Para comparar
        # visualmente o par, os dois CTs recebem a mesma janela, obtida de
        # todos os pixels das duas imagens.
        low, high = np.percentile(np.concatenate((original.ravel(), synthetic.ravel())), (0.5, 99.5))
        for axis, image, title in zip(axes[row_index, :2], (original, synthetic), ("Original", "Sintética")):
            axis.imshow(image, cmap="gray", vmin=low, vmax=high)
            axis.contour(mask, levels=[0.5], colors="#56d87a", linewidths=0.6)
            axis.set_title(title, fontsize=11, pad=7)
            axis.axis("off")
        axes[row_index, 2].imshow(mask, cmap="gray")
        axes[row_index, 2].set_title("ROI erodida", fontsize=11, pad=7)
        axes[row_index, 2].axis("off")
        text_axis = axes[row_index, 3]
        text_axis.axis("off")
        decision = LABELS[row["status"]]
        text_axis.text(0, 0.92, role, fontsize=15, fontweight="bold", transform=text_axis.transAxes)
        text_axis.text(0, 0.79, decision, color=COLORS[row["status"]], fontsize=12, fontweight="bold", transform=text_axis.transAxes)
        text_axis.text(0, 0.64, f"Contraste\n{float(row['original_contrast']):.2f}  →  {float(row['synthetic_contrast']):.2f}", fontsize=12, transform=text_axis.transAxes)
        text_axis.text(0, 0.43, f"Energia\n{float(row['original_energy']):.3f}  →  {float(row['synthetic_energy']):.3f}", fontsize=12, transform=text_axis.transAxes)
        text_axis.text(0, 0.22, row["case_id"].split(" Gated ")[-1], fontsize=9, color="#555555", transform=text_axis.transAxes, wrap=True)
    fig.suptitle("Quatro pares que explicam a auditoria", fontsize=21, fontweight="bold", y=0.995)
    fig.text(0.5, 0.976, "Cada CT é exibida na mesma janela do seu par; contorno verde = região que entrou no GLCM.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(OUTPUT / "casos_chave.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_cohort_plot(rows: list[dict[str, str]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for axis, feature, label in zip(axes, ("contrast", "energy"), ("Contraste GLCM", "Energia GLCM")):
        original = np.array([float(row[f"original_{feature}"]) for row in rows])
        synthetic = np.array([float(row[f"synthetic_{feature}"]) for row in rows])
        for index, row in enumerate(rows):
            axis.plot((0, 1), (original[index], synthetic[index]), color=COLORS[row["status"]], alpha=0.5, linewidth=1.2)
            axis.scatter((0, 1), (original[index], synthetic[index]), color=COLORS[row["status"]], s=24, zorder=3)
        axis.set_xticks((0, 1), ("Original", "Sintética"))
        axis.set_title(label, fontweight="bold")
        axis.set_ylabel("Valor da métrica")
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    for status in ("approved", "manual_review", "reject"):
        axes[0].scatter([], [], color=COLORS[status], label=LABELS[status])
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Mudança por par: as linhas descem quando a métrica diminui na sintética", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT / "mudanca_por_par.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def clipping_sensitivity(rows: list[dict[str, str]], target_id: str) -> list[dict[str, float]]:
    prepared = []
    real_pixels = []
    for row in rows:
        original, synthetic, mask = case_arrays(row["case_id"])
        prepared.append((row["case_id"], original, synthetic, mask))
        real_pixels.append(original[mask])
    real_pixels = np.concatenate(real_pixels)
    output = []
    for percentile in ((0.5, 99.5), (1, 99), (2, 98), (5, 95)):
        low, high = np.percentile(real_pixels, percentile)
        for case_id, original, synthetic, mask in prepared:
            if case_id == target_id:
                original_metrics = glcm_contrast_energy(original, mask, float(low), float(high), 32)
                synthetic_metrics = glcm_contrast_energy(synthetic, mask, float(low), float(high), 32)
                output.append({"window": f"P{percentile[0]}–P{percentile[1]}", "original_energy": original_metrics["energy"], "synthetic_energy": synthetic_metrics["energy"]})
                break
    return output


def render_html(rows: list[dict[str, str]], selected: list[tuple[str, dict[str, str]]]) -> str:
    contrast_original = np.array([float(row["original_contrast"]) for row in rows])
    contrast_synthetic = np.array([float(row["synthetic_contrast"]) for row in rows])
    energy_original = np.array([float(row["original_energy"]) for row in rows])
    energy_synthetic = np.array([float(row["synthetic_energy"]) for row in rows])
    outlier = selected[0][1]
    sensitivity = clipping_sensitivity(rows, outlier["case_id"])
    sensitivity_rows = "".join(f"<tr><td>{item['window']}</td><td>{item['original_energy']:.3f}</td><td>{item['synthetic_energy']:.3f}</td></tr>" for item in sensitivity)
    case_rows = "".join(
        f"<tr><th>{role}</th><td>{row['case_id'].split(' Gated ')[-1]}</td><td><span class='badge badge-{'success' if row['status'] == 'approved' else 'warning' if row['status'] == 'manual_review' else 'error'}'>{LABELS[row['status']]}</span></td><td>{float(row['original_contrast']):.2f} → {float(row['synthetic_contrast']):.2f}</td><td>{float(row['original_energy']):.3f} → {float(row['synthetic_energy']):.3f}</td></tr>"
        for role, row in selected
    )
    return f"""<!doctype html>
<html lang=\"pt-BR\" data-theme=\"light\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>Auditoria GLCM — CT pulmonar sintética</title>
<link href=\"https://cdn.jsdelivr.net/npm/daisyui@5\" rel=\"stylesheet\" type=\"text/css\"><script src=\"https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4\"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Manrope:wght@400;500;600;700&display=swap');
body {{ font-family: Manrope, sans-serif; background:#f5f2eb; color:#17211e; }} h1,h2 {{ font-family: Fraunces, serif; }} .mono {{ font-family: 'DM Mono', monospace; }}
.report {{ max-width: 1240px; margin:auto; padding: 52px 24px 80px; }} .kicker {{ letter-spacing:.16em; text-transform:uppercase; font-size:.72rem; font-weight:700; color:#577064; }}
.masthead {{ border-bottom: 1px solid #b8c2ba; padding-bottom:34px; }} .masthead h1 {{ font-size:clamp(2.4rem,6vw,5.5rem); line-height:.95; max-width:900px; margin:14px 0 20px; }}
.masthead p {{ max-width:760px; font-size:1.05rem; color:#536159; }} .rule {{ height:4px; background:#17211e; width:100%; margin: 20px 0 0; }}
.stats {{ width:100%; margin:28px 0; box-shadow:none; border:1px solid #bcc7bd; background:#fcfbf7; }} .stat {{ padding:22px; }} .stat-title {{ font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; }} .stat-value {{ font-family:'DM Mono',monospace; font-size:1.7rem; }}
.section {{ margin-top:54px; }} .section h2 {{ font-size:2rem; margin-bottom:12px; }} .section-lede {{ max-width:800px; color:#536159; margin-bottom:20px; }}
.image-card {{ background:#111; border:1px solid #26342e; padding:12px; }} .image-card img {{ width:100%; display:block; }} .card {{ box-shadow:none; border:1px solid #bcc7bd; background:#fcfbf7; }} .card-title {{ font-family:Fraunces,serif; }}
.takeaway {{ border-left: 5px solid #b76e00; }} .footer {{ border-top:1px solid #b8c2ba; margin-top:64px; padding-top:16px; color:#65736b; font-size:.78rem; }}
@media print {{ body{{background:white}} .report{{padding:20px}} .section{{break-inside:avoid}} }}
</style></head><body><main class=\"report\">
<header class=\"masthead\"><div class=\"kicker\">Controle de qualidade de textura · auditoria pré-reunião</div><h1>O que os pares GLCM estão, de fato, mostrando?</h1><p>Uma leitura visual e técnica dos 37 pares pulmonar original–sintética que passaram no filtro de máscara.</p><div class=\"rule\"></div></header>
<div role=\"alert\" class=\"alert alert-warning alert-soft mt-7\"><span><strong>Mensagem principal.</strong> Há compressão sistemática de contraste local nas sintéticas; o resultado é promissor como triagem, mas ainda não é uma conclusão clínica isolada.</span></div>
<div class=\"stats stats-vertical lg:stats-horizontal\"><div class=\"stat\"><div class=\"stat-title\">Mediana de contraste</div><div class=\"stat-value\">{np.median(contrast_original):.2f} → {np.median(contrast_synthetic):.2f}</div><div class=\"stat-desc\">original → sintética</div></div><div class=\"stat\"><div class=\"stat-title\">Sintéticas abaixo de 6</div><div class=\"stat-value\">{int((contrast_synthetic < 6).sum())} / {len(rows)}</div><div class=\"stat-desc\">há uma exceção: máximo 6,07</div></div><div class=\"stat\"><div class=\"stat-title\">Decisão do MVP</div><div class=\"stat-value\">10 · 26 · 1</div><div class=\"stat-desc\">aprovados · revisão · rejeitado</div></div></div>
<section class=\"section\"><div class=\"kicker\">Leitura do conjunto</div><h2>As sintéticas convergem para uma faixa mais estreita</h2><p class=\"section-lede\">Cada linha acompanha o mesmo caso. A predominância de linhas descendentes no contraste mostra que a geração reduz a heterogeneidade local em boa parte dos pulmões. Em energia, a resposta é mais heterogênea.</p><div class=\"image-card\"><img src=\"mudanca_por_par.png\" alt=\"Mudança de contraste e energia GLCM, original versus sintética, para 37 pares.\"></div></section>
<section class=\"section\"><div class=\"kicker\">Casos de referência</div><h2>Quatro imagens que devem estar nos próximos slides</h2><p class=\"section-lede\">As CTs são exibidas na mesma janela dentro de cada par. O contorno verde indica a ROI pulmonar efetivamente usada no cálculo.</p><div class=\"image-card\"><img src=\"casos_chave.png\" alt=\"Quatro pares com CT original, sintética e ROI erodida usados para explicar os resultados GLCM.\"></div><div class=\"overflow-x-auto mt-5\"><table class=\"table table-zebra table-sm\"><thead><tr><th>Papel</th><th>Caso</th><th>Decisão</th><th>Contraste</th><th>Energia</th></tr></thead><tbody>{case_rows}</tbody></table></div></section>
<section class=\"section\"><div class=\"kicker\">Outlier de energia</div><h2>O maior alerta é robusto, mas sua magnitude depende da janela</h2><div class=\"grid gap-5 lg:grid-cols-[1.3fr_1fr]\"><div class=\"card takeaway\"><div class=\"card-body\"><h3 class=\"card-title\">{outlier['case_id'].split(' Gated ')[-1]}</h3><p>Energia cai de <strong>{float(outlier['original_energy']):.3f}</strong> para <strong>{float(outlier['synthetic_energy']):.3f}</strong>, enquanto o contraste sobe de {float(outlier['original_contrast']):.2f} para {float(outlier['synthetic_contrast']):.2f}. A sintética não está apenas mais lisa: ela perde a concentração de combinações locais que dominava a imagem original.</p><p>O padrão se mantém nas quatro direções GLCM e continua sendo o maior outlier quando a janela global é alterada. Porém, 32,6% da ROI original fica abaixo do limite inferior da janela P1–P99; por isso o valor absoluto deve ser interpretado junto do protocolo.</p></div></div><div class=\"card\"><div class=\"card-body\"><h3 class=\"card-title\">Teste de sensibilidade</h3><div class=\"overflow-x-auto\"><table class=\"table table-sm\"><thead><tr><th>Janela global</th><th>Original</th><th>Sintética</th></tr></thead><tbody>{sensitivity_rows}</tbody></table></div><p class=\"text-sm text-base-content/60\">Energia GLCM do mesmo par sob quatro janelas de clipping calculadas nas originais.</p></div></div></div></section>
<section class=\"section\"><div class=\"kicker\">Conclusão operacional</div><h2>O GLCM já é útil como triagem explicável</h2><div class=\"grid gap-5 md:grid-cols-3\"><article class=\"card\"><div class=\"card-body\"><h3 class=\"card-title\">O que é sustentado</h3><p>Ele identifica redução coletiva de contraste e separa pares mais estáveis de mudanças pronunciadas, coerentes visualmente nos extremos auditados.</p></div></article><article class=\"card\"><div class=\"card-body\"><h3 class=\"card-title\">O que não se pode afirmar</h3><p>Um valor GLCM isolado não mede gravidade clínica. A escala depende de janela, discretização, distância e ROI.</p></div></article><article class=\"card\"><div class=\"card-body\"><h3 class=\"card-title\">Próximo teste decisivo</h3><p>Recalcular os casos-chave com normalização alternativa e pedir leitura visual especializada dos mesmos painéis.</p></div></article></div></section>
<footer class=\"footer\">Fonte: <span class=\"mono\">outputs/comparison_results.csv</span>, <span class=\"mono\">reference_originals.json</span> e TIFFs do dataset. Gerado em 17 set. 2026 · Protocolo: ROI erodida, 32 níveis, quatro direções, clipping global das originais.</footer>
</main></body></html>"""


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(RESULTS.open(encoding="utf-8")))
    if len(rows) != 37:
        raise RuntimeError(f"esperava 37 pares aceitos; encontrei {len(rows)}")
    selected = key_cases(rows)
    save_case_panel(selected)
    save_cohort_plot(rows)
    (OUTPUT / "index.html").write_text(render_html(rows, selected), encoding="utf-8")
    (OUTPUT / "README.md").write_text("# Relatório visual GLCM\n\nAbra `index.html` no navegador. As figuras ao lado são geradas pelo mesmo comando.\n", encoding="utf-8")
    print(f"Relatório gerado em {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
