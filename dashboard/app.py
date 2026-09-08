"""Dashboard Streamlit: lê CSVs prontos, sem recalcular a pipeline."""
from __future__ import annotations

import csv
import html
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "outputs"
sys.path.insert(0, str(PROJECT))
from src.load_data import load_image  # noqa: E402

FEATURES = {"contrast": "Contraste", "energy": "Energia"}
DECISIONS = {
    "approved": ("Aprovada", "#16803c", "Pode entrar no conjunto de data augmentation."),
    "approved_with_observation": ("Aprovada com observação", "#277da1", "Caso raro, mas compatível com a original."),
    "manual_review": ("Revisão manual", "#b76e00", "Ainda não entra no augmentation: vale olhar este par com calma."),
    "reject": ("Não aprovada", "#c73e1d", "Não deve entrar no data augmentation."),
}


def read_csv(name: str) -> list[dict]:
    with (OUTPUT / name).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def as_float(value: str | float) -> float:
    return float(value)


def normalised(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, (1, 99))
    return np.clip((image - low) / max(high - low, 1e-12), 0, 1)


def percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def plain_number(value: float) -> str:
    return f"{value:.4f}" if abs(value) < 0.1 else f"{value:.2f}"


def decision_card(result: dict) -> None:
    title, colour, meaning = DECISIONS[result["status"]]
    st.markdown(
        f"""<div style="border-left:7px solid {colour};background:#f8f9fb;border-radius:8px;padding:16px 18px;margin:4px 0 18px">
        <div style="font-size:1.15rem;font-weight:700;color:{colour}">{title}</div>
        <div style="margin-top:5px">{html.escape(meaning)}</div>
        <div style="margin-top:7px;color:#555">Motivo: {html.escape(result["decision_reason"])}</div></div>""",
        unsafe_allow_html=True,
    )


def metric_interpretation(feature: str, result: dict, stats: dict) -> None:
    original = as_float(result[f"original_{feature}"])
    synthetic = as_float(result[f"synthetic_{feature}"])
    relative = as_float(result[f"relative_delta_{feature}"])
    iqr_distance = as_float(result[f"pair_distance_{feature}_iqr"])
    outside = result[f"synthetic_{feature}_outside_reference"] == "True"
    direction = "maior" if synthetic > original else "menor"
    reference_text = "fora" if outside else "dentro"
    st.markdown(f"**Comparação com o par:** a sintética ficou **{percentage(relative)} {direction}** que a original.")
    st.caption(
        f"Isso equivale a {iqr_distance:.2f}× a variação central das originais (IQR). "
        f"O alerta do MVP começa acima de {as_float(result['pair_iqr_threshold']):.1f}× IQR."
    )
    st.markdown(
        f"**Comparação com a referência:** o valor sintético está **{reference_text}** da faixa esperada "
        f"({plain_number(as_float(stats['tukey_lower']))} a {plain_number(as_float(stats['tukey_upper']))})."
    )


def reference_figure(feature: str, result: dict, stats: dict):
    """Faixa da referência, mediana e os dois valores do par em uma só leitura."""
    original = as_float(result[f"original_{feature}"])
    synthetic = as_float(result[f"synthetic_{feature}"])
    lower, upper = as_float(stats["tukey_lower"]), as_float(stats["tukey_upper"])
    median = as_float(stats["median"])
    spread = max(upper - lower, abs(original - synthetic), 1e-9)
    x_min = min(lower, original, synthetic) - 0.18 * spread
    x_max = max(upper, original, synthetic) + 0.18 * spread
    fig, ax = plt.subplots(figsize=(8, 1.75))
    ax.axvspan(lower, upper, color="#dff3e4", zorder=0)
    ax.hlines(0, x_min, x_max, color="#a0a0a0", linewidth=1.2, zorder=1)
    ax.vlines(median, -0.3, 0.3, color="#277da1", linestyle="--", linewidth=1.4, label="Mediana das reais")
    ax.scatter([original], [0], color="#2f5d8a", s=80, zorder=3, label="Original do par")
    ax.scatter([synthetic], [0], color="#ed7d31", s=80, zorder=3, label="Sintética")
    ax.set_xlim(x_min, x_max)
    ax.set_yticks([])
    ax.set_xlabel("Valor da feature")
    ax.set_title("Faixa verde = região esperada nas CTs reais validadas", fontsize=10, loc="left")
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.38), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def compact_table(rows: list[dict], columns: list[tuple[str, str]]) -> list[dict]:
    return [{label: row.get(key, "") for key, label in columns} for row in rows]


st.set_page_config(page_title="Validação GLCM pulmonar", layout="wide")
st.title("Validação de textura pulmonar")
st.caption("GLCM · Contraste e energia · O painel só interpreta resultados já calculados.")
required = ["case_validation.csv", "features_glcm.csv", "reference_originals.csv", "comparison_results.csv"]
if not all((OUTPUT / name).exists() for name in required):
    st.error("Ainda não há resultados. Execute a pipeline antes de abrir o painel.")
    st.code("python -m src.run_pipeline --data-dir /caminho/para/o/dataset", language="bash")
    st.stop()

validation, features, reference, comparisons = (read_csv(name) for name in required)
accepted = [row for row in validation if row["status"] == "accepted"]
reference_by_feature = {row["feature"]: row for row in reference}
result_by_case = {row["case_id"]: row for row in comparisons}
approved_count = sum(row["status"] in {"approved", "approved_with_observation"} for row in comparisons)
review_count = sum(row["status"] == "manual_review" for row in comparisons)
reject_count = sum(row["status"] == "reject" for row in comparisons)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Pares utilizáveis", len(accepted), help="Imagem e máscara passaram na validação inicial.")
col2.metric("Prontas para usar", approved_count, help="Sintéticas aprovadas para augmentation neste MVP.")
col3.metric("Pedem revisão", review_count, help="Não entram no augmentation até revisão humana.")
col4.metric("Não aprovadas", reject_count, help="Distantes do par e fora da faixa das originais reais.")

st.divider()
st.header("Leia uma imagem sintética", anchor=False)
st.write("Escolha um par. Veja as imagens e, depois, as duas perguntas que o framework responde.")
case_id = st.selectbox("Par original–sintética", [row["case_id"] for row in accepted])
case = next(row for row in accepted if row["case_id"] == case_id)
result = result_by_case[case_id]
original, synthetic, mask = (load_image(Path(case[key])) for key in ("original_path", "synthetic_path", "mask_path"))

images = st.columns(3)
images[0].image(normalised(original), caption="1. CT original que condicionou a geração", clamp=True)
images[1].image(normalised(synthetic), caption="2. CT sintética que queremos avaliar", clamp=True)
images[2].image((np.isclose(mask, 1.0)).astype(np.uint8) * 255, caption="3. Máscara: região usada para medir a textura", clamp=True)
decision_card(result)

st.subheader("As duas comparações", anchor=False)
st.caption("Uma imagem só é rejeitada automaticamente se falhar nas duas perguntas abaixo.")
feature_columns = st.columns(2)
for column, (feature, label) in zip(feature_columns, FEATURES.items()):
    with column:
        st.markdown(f"### {label}")
        st.pyplot(reference_figure(feature, result, reference_by_feature[feature]), clear_figure=True)
        metric_interpretation(feature, result, reference_by_feature[feature])

st.info("Como ler: o percentual mede a mudança em relação ao próprio par. A faixa verde mostra se a sintética ainda parece compatível com as CTs reais validadas.")
with st.expander("Como o MVP toma a decisão?", expanded=False):
    st.markdown(
        "- **Aprovada:** próxima da original do par e dentro da faixa das reais.\n"
        "- **Revisão manual:** mudou bastante em relação ao par, ou saiu da faixa real sem preservar a raridade da original.\n"
        "- **Não aprovada:** mudou bastante em relação ao par **e** também ficou fora da faixa das reais.\n"
        "- **Aprovada com observação:** par original e sintética parecem um caso raro de forma compatível."
    )

st.divider()
st.header("Visão do conjunto", anchor=False)
left, right = st.columns(2)
figures = OUTPUT / "figures"
if (figures / "reference_distributions.png").exists():
    left.image(str(figures / "reference_distributions.png"), caption="Originais aceitas: média (vermelho), mediana (verde) e faixa operacional (tracejado).")
if (figures / "original_vs_synthetic.png").exists():
    right.image(str(figures / "original_vs_synthetic.png"), caption="Quanto mais perto da diagonal, mais a sintética preserva a feature da original do mesmo par.")

st.subheader("Resumo de todos os pares", anchor=False)
summary_rows = []
for row in comparisons:
    title, _, _ = DECISIONS[row["status"]]
    summary_rows.append({
        "Caso": row["case_id"], "Decisão": title,
        "Contraste: distância do par": percentage(as_float(row["relative_delta_contrast"])),
        "Contraste: referência": "Fora" if row["synthetic_contrast_outside_reference"] == "True" else "Dentro",
        "Energia: distância do par": percentage(as_float(row["relative_delta_energy"])),
        "Energia: referência": "Fora" if row["synthetic_energy_outside_reference"] == "True" else "Dentro",
        "Explicação": row["decision_reason"],
    })
st.dataframe(summary_rows, use_container_width=True, hide_index=True)

with st.expander("Dados técnicos e validação das imagens", expanded=False):
    st.markdown("**Casos aceitos e descartados**")
    st.dataframe(compact_table(validation, [("case_id", "Caso"), ("status", "Status"), ("reason", "Motivo")]), use_container_width=True, hide_index=True)
    st.markdown("**Referência global das originais**")
    st.dataframe(reference, use_container_width=True, hide_index=True)
    st.markdown("**Todas as features GLCM calculadas**")
    st.dataframe(features, use_container_width=True, hide_index=True)
