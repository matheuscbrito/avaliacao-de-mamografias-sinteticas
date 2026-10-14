"""Relatório visual simples para os resultados do MVP."""
from __future__ import annotations
import html
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

METRICS = (
    ("glcm_contrast", "GLCM — contraste", "Detalhe entre pixels vizinhos; menor sugere suavização."),
    ("glcm_energy", "GLCM — energia", "Concentração dos pares de tons."),
    ("wavelet_energy", "Wavelet — energia", "Quantidade de detalhe nas sub-bandas."),
    ("wavelet_entropy", "Wavelet — entropia", "Variedade dos detalhes wavelet."),
    ("wavelet_ll", "Wavelet — LL", "Componente de baixa frequência: leitura complementar de brilho."),
)

def _save_overview(rows: list[dict], output: Path) -> None:
    counts = {key: sum(r["brightness_status"] == key for r in rows) for key in ("approved", "manual_review", "hard_reject")}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ("#54A24B", "#F2A541", "#E45756")
    axes[0].bar(("Aprovada", "Revisão", "Rejeição"), counts.values(), color=colors)
    axes[0].set(title="Filtro de brilho", ylabel="Número de cortes")
    axes[0].text(.5, -.23, "Decisão: Δμ e saturação no tom 31.\nLL é exibida como leitura complementar.", transform=axes[0].transAxes, ha="center", fontsize=9)
    axes[1].scatter([r["delta_mu"] for r in rows], [r["sat31_pct"] for r in rows], c=[colors[("approved", "manual_review", "hard_reject").index(r["brightness_status"])] for r in rows])
    axes[1].axhline(15, color="#E45756", linestyle="--", label="Sat31 = 15%")
    axes[1].set(title="Brilho por corte", xlabel="Δμ: sintética − real", ylabel="Sat31 (%)"); axes[1].legend()
    fig.tight_layout(); fig.savefig(output, dpi=170, bbox_inches="tight"); plt.close(fig)

def _save_pairs(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8)); axes = axes.ravel()
    for axis, (key, title, _) in zip(axes, METRICS):
        x=np.array([float(r[f"{key}_real"]) for r in rows]); y=np.array([float(r[f"{key}_synthetic"]) for r in rows])
        lo, hi=min(x.min(),y.min()), max(x.max(),y.max()); pad=max((hi-lo)*.06, 1e-8)
        axis.scatter(x,y,color="#4C78A8",alpha=.85); axis.plot([lo-pad,hi+pad],[lo-pad,hi+pad],"--",color="#777")
        axis.set(title=title, xlabel="Real", ylabel="Sintética")
    axes[-1].axis("off"); axes[-1].text(0, .8, "Como ler\n\nCada ponto é um corte.\nA diagonal significa igualdade.\nAfastamento sistemático indica diferença entre real e sintética.", fontsize=11, va="top")
    fig.tight_layout(); fig.savefig(output, dpi=170, bbox_inches="tight"); plt.close(fig)

def generate_report(rows: list[dict], output_dir: Path) -> None:
    report = output_dir / "report"; report.mkdir(exist_ok=True)
    _save_overview(rows, report / "01_brilho.png"); _save_pairs(rows, report / "02_metricas_pareadas.png")
    cards=[]
    for n, row in enumerate(rows):
        values="<br>".join(f"<b>{html.escape(title)}</b>: real {float(row[key+'_real']):.4g} · sintética {float(row[key+'_synthetic']):.4g}" for key,title,_ in METRICS)
        cards.append(f"<article><img src='../previews/{n:02d}.png'><h3>{html.escape(row['case_id'])}</h3><p><b>Brilho:</b> {row['brightness_status']} · Δμ={float(row['delta_mu']):.3f} · Sat31={float(row['sat31_pct']):.1f}%</p><p>{values}</p></article>")
    descriptions="".join(f"<li><b>{title}:</b> {text}</li>" for _,title,text in METRICS)
    (report / "index.html").write_text(f"""<!doctype html><html lang='pt-BR'><meta charset='utf-8'><title>Relatório MVP — CT sintética</title><style>body{{font:16px system-ui;margin:32px;color:#17202a;background:#f8fafc}}main{{max-width:1300px;margin:auto}}img{{max-width:100%;background:white}}.charts{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px}}article{{background:#fff;padding:14px;border:1px solid #dce3ea;border-radius:8px}}h1{{margin-bottom:4px}}p{{line-height:1.45}}</style><main><h1>Relatório visual — CT pulmonar sintética</h1><p>Comparação pareada: cada imagem sintética é lida contra sua CT real correspondente, dentro da ROI pulmonar erodida.</p><div class='charts'><img src='01_brilho.png'><img src='02_metricas_pareadas.png'></div><h2>O que significa cada número</h2><ul>{descriptions}</ul><h2>Casos analisados</h2><div class='grid'>{''.join(cards)}</div></main></html>""",encoding="utf-8")
