"""Página Dgen × Dsr: lê os resultados de `src.compare_generations`, sem recalcular nada."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT.parent / "resultados" / "comparacao_Dgen_Dsr_wavelet"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "dashboard"))
from common import FEATURES, DECISIONS, as_float, decision_card, percentage, plain_number, read_csv  # noqa: E402
from src.load_data import load_image  # noqa: E402

GENERATIONS = {"dgen": "Dgen", "dsr": "Dsr"}
COLORS = {"original": "#333333", "dgen": "#E45756", "dsr": "#4C78A8"}
BANDS = ("LH", "HL", "HH")


@st.cache_data
def load_case_images(case_id: str, dgen_dir: str, dsr_dir: str) -> tuple[np.ndarray, ...]:
    dgen, dsr = RESULTS / dgen_dir, RESULTS / dsr_dir
    return (
        load_image(dgen / f"{case_id}.tiff"),
        load_image(dgen / f"{case_id}_generate.tiff"),
        load_image(dsr / f"{case_id}_generate.tiff"),
        load_image(dgen / f"{case_id}_mask.tiff"),
    )


def images_figure(original: np.ndarray, dgen: np.ndarray, dsr: np.ndarray, mask: np.ndarray):
    """Mesma janela de intensidade nas três CTs; contorno da máscara e mapa de erro como na comparação GLCM."""
    low, high = np.percentile(original, (1, 99))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for axis, image, title in zip(axes[:3], (original, dgen, dsr), ("Original", "Dgen", "Dsr")):
        axis.imshow(image, cmap="gray", vmin=low, vmax=high)
        axis.contour(mask, levels=[0.5], colors="#00c980", linewidths=.7)
        axis.set_title(title)
    error = np.abs(dsr - original) - np.abs(dgen - original)
    limit = max(float(np.percentile(np.abs(error), 99)), 1e-9)
    axes[3].imshow(error, cmap="coolwarm", vmin=-limit, vmax=limit)
    axes[3].set_title("Erro Dsr − erro Dgen\n(azul favorece Dsr)")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    return fig


def reference_figure(feature: str, row: dict, stats: dict):
    """Faixa da referência, mediana, a original e as duas sintéticas em uma só leitura."""
    values = {key: as_float(row[f"{key}_{feature}"]) for key in ("original", *GENERATIONS)}
    lower, upper = as_float(stats["tukey_lower"]), as_float(stats["tukey_upper"])
    spread = max(upper - lower, max(values.values()) - min(values.values()), 1e-9)
    x_min = min(lower, *values.values()) - 0.18 * spread
    x_max = max(upper, *values.values()) + 0.18 * spread
    fig, ax = plt.subplots(figsize=(8, 1.9))
    ax.axvspan(lower, upper, color="#dff3e4", zorder=0)
    ax.hlines(0, x_min, x_max, color="#a0a0a0", linewidth=1.2, zorder=1)
    ax.vlines(as_float(stats["median"]), -0.3, 0.3, color="#277da1", linestyle="--", linewidth=1.4, label="Mediana das reais")
    for key, marker in (("original", "D"), ("dgen", "o"), ("dsr", "o")):
        ax.scatter([values[key]], [0], color=COLORS[key], marker=marker, s=80, zorder=3, label=GENERATIONS.get(key, "Original do par"))
    ax.set_xlim(x_min, x_max)
    ax.set_yticks([])
    ax.set_title("Faixa verde = região esperada nas CTs reais validadas", fontsize=10, loc="left")
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=4, frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def subband_figure(row: dict, levels: int):
    labels = [f"{band} e{scale}" for scale in range(1, levels + 1) for band in BANDS]
    positions = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 2.8))
    for offset, (key, name) in zip((-0.2, 0.2), GENERATIONS.items()):
        ratios = [as_float(row[f"log2_energy_ratio_{key}_s{scale}_{band}"]) for scale in range(1, levels + 1) for band in BANDS]
        ax.bar(positions + offset, ratios, width=0.4, color=COLORS[key], label=name)
    ax.axhline(0, color="#888888", linestyle="--", linewidth=1)
    ax.set_xticks(positions, labels, fontsize=8)
    ax.set_ylabel("log2(sintética / original)", fontsize=8)
    ax.set_title("Energia por sub-banda (e1 = escala mais fina); abaixo de 0 = detalhe perdido", fontsize=10, loc="left")
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def feature_interpretation(feature: str, row: dict) -> None:
    original = as_float(row[f"original_{feature}"])
    for key, name in GENERATIONS.items():
        synthetic = as_float(row[f"{key}_{feature}"])
        direction = "maior" if synthetic > original else "menor"
        outside = "fora" if row[f"{key}_{feature}_outside_reference"] == "True" else "dentro"
        st.markdown(
            f"**{name}:** {percentage(as_float(row[f'{key}_relative_delta_{feature}']))} {direction} que a original "
            f"· {as_float(row[f'{key}_pair_distance_{feature}_iqr']):.2f}× IQR · {outside} da faixa real"
        )


st.set_page_config(page_title="Wavelet: Dgen × Dsr", layout="wide")
st.title("Comparação Dgen × Dsr")
st.caption("Wavelet de imagem inteira · Mesmos cortes, mesmas originais e mesmas máscaras nas duas gerações · O painel só lê resultados já calculados.")
required = ["resumo.json", "metricas_por_caso.csv", "referencia_originais.csv"]
if not all((RESULTS / name).exists() for name in required):
    st.error("Ainda não há resultados da comparação. Rode, a partir de `wavelet/`:")
    st.code("python -m src.compare_generations --dgen-dir ../dados/Dgen --dsr-dir ../dados/Dsr --output-dir ../resultados/comparacao_Dgen_Dsr_wavelet", language="bash")
    st.stop()

summary = json.loads((RESULTS / "resumo.json").read_text(encoding="utf-8"))
rows = read_csv(RESULTS / "metricas_por_caso.csv")
reference_by_feature = {row["feature"]: row for row in read_csv(RESULTS / "referencia_originais.csv")}
total = summary["valid_pairs"]

status_counts = summary["status_counts"]
energy_ratio = {key: np.median([as_float(r[f"{key}_energy"]) / as_float(r["original_energy"]) for r in rows]) for key in GENERATIONS}
overview = [
    {"Métrica": f"{label}: pares mais próximos da original", **{name: f"{summary[feature][f'{key}_closer']} de {total}" for key, name in GENERATIONS.items()},
     "Leitura": f"Wilcoxon pareado p = {summary[feature]['wilcoxon_p']:.1g}"}
    for feature, label in FEATURES.items()
] + [
    {"Métrica": f"{label}: correlação com a original (Spearman)", **{name: f"{summary[feature][f'spearman_original_{key}']:.2f}" for key, name in GENERATIONS.items()},
     "Leitura": "perto de 1 = preserva quais cortes têm mais ou menos textura"}
    for feature, label in FEATURES.items()
] + [
    {"Métrica": "Energia sintética / original (mediana)", **{name: f"{energy_ratio[key]:.0%}" for key, name in GENERATIONS.items()},
     "Leitura": "abaixo de 100% = detalhe de alta frequência perdido"},
    {"Métrica": "Aprovadas pelo framework", **{name: f"{status_counts[key]['approved'] + status_counts[key]['approved_with_observation']} de {total}" for key, name in GENERATIONS.items()},
     "Leitura": "o erro de energia passa de 1 IQR nas duas gerações"},
]
st.dataframe(overview, use_container_width=True, hide_index=True)
st.info(
    "Leitura rápida: o Dsr segue a textura de cada corte e o Dgen não, mas as duas gerações ainda suavizam o parênquima. "
    "Análise completa em `resultados/comparacao_Dgen_Dsr_wavelet/resumo.md`."
)

st.divider()
st.header("Compare um corte", anchor=False)
order = st.radio(
    "Ordenar cortes por",
    ("Nome", "Maior vantagem do Dsr", "Maior vantagem do Dgen"),
    horizontal=True,
    help="Vantagem = diferença entre as distâncias ao par (em IQR), somando energia e entropia.",
)


def advantage(row: dict) -> float:
    return sum(as_float(row[f"dgen_pair_distance_{f}_iqr"]) - as_float(row[f"dsr_pair_distance_{f}_iqr"]) for f in FEATURES)


ordered = {"Nome": rows, "Maior vantagem do Dsr": sorted(rows, key=advantage, reverse=True), "Maior vantagem do Dgen": sorted(rows, key=advantage)}[order]
case_id = st.selectbox("Corte", [row["case_id"] for row in ordered])
row = next(row for row in rows if row["case_id"] == case_id)

original, dgen, dsr, mask = load_case_images(case_id, summary["dgen_dir"], summary["dsr_dir"])
st.pyplot(images_figure(original, dgen, dsr, mask), clear_figure=True)

cards = st.columns(2)
for column, (key, name) in zip(cards, GENERATIONS.items()):
    with column:
        decision_card(row[f"{key}_status"], row[f"{key}_decision_reason"], heading=name)

feature_columns = st.columns(2)
for column, (feature, label) in zip(feature_columns, FEATURES.items()):
    with column:
        st.markdown(f"### {label}")
        st.pyplot(reference_figure(feature, row, reference_by_feature[feature]), clear_figure=True)
        feature_interpretation(feature, row)
chart, note = st.columns([3, 2])
chart.pyplot(subband_figure(row, int(summary["level"])), clear_figure=True)
note.markdown(
    "**Em qual escala o detalhe muda?** Cada barra compara a energia de uma sub-banda da sintética com a da original. "
    "LH, HL e HH são as direções horizontal, vertical e diagonal; e1 é a escala mais fina (bordas de 1–2 pixels), e2 a seguinte."
)
note.caption("Barras abaixo de 0 indicam perda de detalhe; −1 equivale a metade da energia original.")

st.divider()
st.header("Visão do conjunto", anchor=False)
left, right = st.columns(2)
figures = RESULTS / "figuras"
left.image(str(figures / "original_vs_sinteticas.png"), caption="Cada ponto é um corte: o Dsr segue a diagonal; a energia do Dgen fica numa faixa horizontal, sem acompanhar a original.")
right.image(str(figures / "energia_por_subbanda.png"), caption="As duas gerações perdem energia em todas as sub-bandas; no Dsr a perda é mais uniforme entre cortes.")

st.subheader("Resumo de todos os cortes", anchor=False)
table = []
for item in ordered:
    entry = {"Corte": item["case_id"]}
    for key, name in GENERATIONS.items():
        entry[f"Decisão {name}"] = DECISIONS[item[f"{key}_status"]][0]
    for feature, label in FEATURES.items():
        for key, name in GENERATIONS.items():
            entry[f"{label} {name} (IQR)"] = round(as_float(item[f"{key}_pair_distance_{feature}_iqr"]), 2)
    table.append(entry)
st.dataframe(table, use_container_width=True, hide_index=True)

with st.expander("Parâmetros e referência", expanded=False):
    st.markdown(
        f"- Wavelet `{summary['wavelet']}`, {summary['level']} níveis, fração mínima de tecido {summary['min_tissue']}\n"
        f"- {summary['valid_pairs']} cortes válidos de {summary['shared_pairs']}; limiar do par: {summary['pair_iqr_threshold']}× IQR\n"
        f"- Clipping calculado nas originais: {plain_number(summary['clip_interval'][0])} a {plain_number(summary['clip_interval'][1])}"
    )
    st.dataframe(list(reference_by_feature.values()), use_container_width=True, hide_index=True)
