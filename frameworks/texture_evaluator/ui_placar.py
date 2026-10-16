"""Painel científico para inspeção de CT pulmonar sintética por dataset."""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from streamlit.errors import StreamlitAPIException

from frameworks.texture_evaluator.placar import PlacarError, add_version, build_board, load_case, load_history, refresh_history, remove_version, technique_table, write_outputs
from frameworks.texture_evaluator.decision import BRIGHTNESS_TECHNIQUES, GLCM_METRICS, METRIC_STATUSES, NEX_STATUS, TECHNIQUES, WAVELET_METRICS, brightness_statuses, brightness_verdict, technique_statuses, texture_status
from frameworks.texture_evaluator.glcm.src import quantize
from frameworks.texture_evaluator.regional import MASK_ONLY, PAIR_MIN_CORRELATION
from frameworks.texture_evaluator.run_mvp import context_from_dict
from frameworks.texture_evaluator.visual import MAP_LAYERS, TECHNIQUE_FIGURES, map_figure, technique_figure


# Atalhos de navegação; cada dataset mantém histórico e calibração próprios.
DATASETS = {
    "Dsr": {"folder": "dados/Dsr", "masks": "dados/Dgen", "history": "outputs/placar", "description": "Sintética gerada da própria CT em baixa resolução; comparação ponto a ponto é válida."},
    "Dgen": {"folder": "dados/Dgen", "masks": "dados/Dgen", "history": "outputs/placar_dgen", "description": "Sintética gerada a partir do contorno pulmonar; a comparação é global, não ponto a ponto."},
}
# limites do filtro de brilho: revisão acima de 1 IQR das médias pulmonares das originais, rejeição acima de 2;
# lidos da calibração do histórico (ver `calibration_limits`), para a UI nunca divergir do framework
OFFICIAL_REVIEW, OFFICIAL_REJECT, OFFICIAL_SATURATION = float("nan"), float("nan"), 15.0
STATUS_TEXT = {"approved": "aprovada", "manual_review": "revisão manual", "hard_reject": "rejeição direta"}
DENSITY_TEXT = {"ok": "normal", "brighter": "clareou", "darker": "escureceu", "too_bright": "claro demais", "too_dark": "escuro demais"}
FINAL_TEXT = {"aprovada": "aprovada", "revisar": "revisão necessária", "rejeitar": "rejeição direta"}
NEX_TEXT = {"ok": "dentro da faixa real", "alto": "acima da faixa real", "baixo": "abaixo da faixa real", "inconclusivo": "inconclusivo (poucas janelas)"}


def resolve(text: str) -> Path:
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def short(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def case_name(case_id: str) -> str:
    phase = re.search(r"Gated ([\d.]+A)", case_id)
    return f"{phase.group(1)} · {case_id.rsplit('-', 1)[-1]}" if phase else case_id


def png(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=135, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buffer.getvalue()


def number(row: dict, column: str) -> float:
    return float(row[column])


def br(value: float) -> str:
    """Número com vírgula decimal, como no resto do texto."""
    return f"{value:.3f}".replace(".", ",")


def calibration_limits(history: dict) -> None:
    global OFFICIAL_REVIEW, OFFICIAL_REJECT
    OFFICIAL_REVIEW = history["calibration"]["mean_iqr"]
    OFFICIAL_REJECT = 2 * OFFICIAL_REVIEW


BRIGHTNESS_LABEL = {"aprovada": "aprovada", "revisão": "revisão manual", "reprovada": "rejeição direta"}


def brightness_decision(row: dict) -> str:
    """Veredito do filtro de brilho: o mais severo entre GLCM (Δμ, Sat31) e Wavelet (LL)."""
    return BRIGHTNESS_LABEL[brightness_verdict(row)]


def glcm_brightness_why(row: dict) -> str:
    """Por que a GLCM (régua de 32 tons: Δμ e Sat31) classificou o brilho assim."""
    delta, saturation = number(row, "abs_delta_mu"), number(row, "sat31_pct")
    if saturation > OFFICIAL_SATURATION:
        return f"Sat31 = {saturation:.1f}% passou de 15%: pixels demais no tom mais claro.".replace(".", ",", 1)
    if delta > OFFICIAL_REJECT:
        return f"|Δμ| = {br(delta)} passou de {br(OFFICIAL_REJECT)}."
    if delta > OFFICIAL_REVIEW:
        return f"|Δμ| = {br(delta)} ficou entre {br(OFFICIAL_REVIEW)} e {br(OFFICIAL_REJECT)}."
    return f"|Δμ| = {br(delta)} e Sat31 = {saturation:.1f}% dentro dos limites.".replace(f"{saturation:.1f}", f"{saturation:.1f}".replace(".", ","))


def wavelet_brightness_why(row: dict) -> str:
    """Por que a Wavelet (média da banda LL) classificou o brilho assim."""
    z, status = number(row, "wavelet_ll_z"), row["wavelet_ll_status"]
    side = "mais clara" if z > 0 else "mais escura"
    if status == "aprovada":
        return f"LL da sintética a {f'{z:+.2f}'.replace('.', ',')} IQR da real: dentro de ±1 IQR."
    limit = "±2 IQR: fora da variação entre CTs reais" if status == "reprovada" else "±1 IQR, mas ficou dentro de ±2 IQR"
    return f"Sintética {side}: LL a {f'{z:+.2f}'.replace('.', ',')} IQR da real, passou de {limit}."


def brightness_explanation(row: dict) -> str:
    """Motivo do veredito de brilho, citando só as técnicas fora da faixa."""
    statuses = brightness_statuses(row)
    why = {"GLCM": glcm_brightness_why, "Wavelet": wavelet_brightness_why}
    reasons = [f"{technique}: {why[technique](row)}" for technique, status in statuses.items() if status != "aprovada"]
    return " · ".join(reasons) if reasons else "GLCM e Wavelet LL dentro dos limites"


def density_reading(row: dict) -> str:
    """Densidade: pixels fora da faixa P1–P99 real e deslocamento da mediana do pulmão (em IQR das originais)."""
    return f"{DENSITY_TEXT[row['density_status']]} (mediana {number(row, 'density_median_shift_iqr'):+.2f} IQR)".replace(".", ",")


def ll_shift(row: dict) -> float:
    return abs(number(row, "wavelet_ll_synthetic") - number(row, "wavelet_ll_real"))


def final_status(row: dict) -> str:
    return FINAL_TEXT[row["final_status"]]


def nex_reading(row: dict) -> str:
    return f"{NEX_TEXT[row['nex_status']]} (Δ {number(row, 'nex_delta'):+.3f})".replace(".", ",")


TEXTURE_LABEL = {"aprovada": "textura preservada", "revisão": "revisão necessária", "reprovada": "reprovada"}
TECHNIQUE_METRICS = {"GLCM": ("glcm", GLCM_METRICS), "Wavelet": ("wavelet", WAVELET_METRICS)}
STATUS_COLORS = alt.Scale(domain=[*METRIC_STATUSES, "inconclusivo"], range=["#0b6b32", "#d9a21b", "#9f1f18", "#b8bdbb"])


def metric_reading(row: dict, prefix: str, key: str) -> str:
    """Classificação de uma métrica e a diferença em IQR das CTs reais (ex.: revisão (−1,4 IQR))."""
    return f"{row[f'{prefix}_{key}_status']} ({number(row, f'{prefix}_{key}_z'):+.1f} IQR)".replace(".", ",")


def wavelet_metric_reading(row: dict, key: str) -> str:
    return metric_reading(row, "wavelet", key)


def technique_reason(row: dict, technique: str) -> str:
    """O que a técnica disse neste corte, só com as métricas fora da faixa."""
    if technique == "NEx":
        return "" if row["nex_status"] == "ok" else f"{NEX_TEXT[row['nex_status']]} (Δ {number(row, 'nex_delta'):+.3f})".replace(".", ",")
    prefix, metrics = TECHNIQUE_METRICS[technique]
    flagged = [f"{label.removeprefix(technique + ' ')} {metric_reading(row, prefix, key)}" for key, label in metrics.items() if row[f"{prefix}_{key}_status"] != "aprovada"]
    verdict = row.get("wavelet_verdict", "textura preservada")
    if technique == "Wavelet" and flagged and verdict != "textura preservada":
        flagged.append(f"tipo: {verdict}")
    return ", ".join(flagged)


def texture_decision(row: dict) -> tuple[str, str]:
    """Veredito da aba de textura, deliberadamente independente do brilho.

    GLCM, Wavelet e NEx são classificadas cada uma por conta própria; a pior
    delas decide. GLCM e Wavelet usam a regra do filtro de brilho (diferença
    em IQR das CTs reais: revisão > 1, reprovada > 2); NEx fora de P5–P95 das
    CTs reais pede revisão.
    """
    reasons = [f"{technique}: {reason}" for technique in TECHNIQUES if (reason := technique_reason(row, technique))]
    return TEXTURE_LABEL[texture_status(row)], " · ".join(reasons) if reasons else "GLCM, Wavelet e NEx dentro das faixas de referência"


def texture_explanation(row: dict) -> str:
    energy = 100 * number(row, "glcm_energy_synthetic") / number(row, "glcm_energy_real")
    text = f"GLCM contraste {number(row, 'glcm_contrast_pct'):.0f}%; energia GLCM {energy:.0f}%; Wavelet 2 px {number(row, 'wavelet_fine_pct'):.0f}%; Wavelet 4 px {number(row, 'wavelet_coarse_pct'):.0f}%"
    verdict = row.get("wavelet_verdict", "")
    return text + (f" · leitura exploratória: {verdict}" if verdict and verdict != "textura preservada" else "")


STATUS_BOX = {"aprovada": (st.success, ":material/check_circle:"), "revisão": (st.warning, ":material/visibility:"),
              "reprovada": (st.error, ":material/cancel:"), "inconclusivo": (st.info, ":material/help:")}
STATUS_MARK = {"aprovada": "✓", "revisão": "!", "reprovada": "✕", "inconclusivo": "?"}


def metric_line(row: dict, prefix: str, key: str, label: str) -> str:
    """Uma métrica em linguagem direta: classificação, diferença em IQR e quanto a sintética vale da real."""
    z, status = number(row, f"{prefix}_{key}_z"), row[f"{prefix}_{key}_status"]
    pct = 100 * number(row, f"{prefix}_{key}_synthetic") / number(row, f"{prefix}_{key}_real")
    return f"{STATUS_MARK[status]} **{label}**: Δ {z:+.1f} IQR → {status} · sintética = {pct:.0f}% da real".replace(".", ",")


def technique_why(row: dict, technique: str, reference: dict) -> str:
    """Por que a técnica chegou à classificação deste corte, citando o limite que foi cruzado."""
    status = technique_statuses(row)[technique]
    if technique == "NEx":
        value, band = number(row, "nex_synthetic"), f"{br(reference['p5'])} a {br(reference['p95'])}"
        if status == "inconclusivo":
            return "Poucas janelas de 32 px dentro do pulmão: leitura inconclusiva, não pesa no veredito."
        side = {"ok": "dentro", "alto": "acima", "baixo": "abaixo"}[row["nex_status"]]
        return f"NEx da sintética = {br(value)}, {side} da faixa das CTs reais ({band})."
    prefix, metrics = TECHNIQUE_METRICS[technique]
    names = [label.removeprefix(technique + " ") for key, label in metrics.items() if row[f"{prefix}_{key}_status"] == status]
    if status == "aprovada":
        return "Todas as métricas ficaram dentro de ±1 IQR das CTs reais."
    joined = " e ".join(names)
    if status == "revisão":
        return f"{joined} passou de ±1 IQR, mas ficou dentro de ±2 IQR."
    return f"{joined} passou de ±2 IQR: fora da variação entre CTs reais."


def verdict_summary(statuses: dict, verdict: str) -> str:
    """Frase do veredito combinado: quem levou o corte a essa classificação."""
    if verdict == "aprovada":
        return f"{', '.join(list(statuses)[:-1])} e {list(statuses)[-1]} aprovaram."
    who = [technique for technique, status in statuses.items() if status == verdict]
    verb = "reprovou" if verdict == "reprovada" else "pediu revisão"
    if len(who) > 1:
        verb = "reprovaram" if verdict == "reprovada" else "pediram revisão"
    return f"{' e '.join(who)} {verb}. O veredito é sempre o da técnica mais severa."


def show_texture_limits(reference: dict):
    """Régua única dos limites de textura, sempre visível."""
    nex_band = f"{br(reference['p5'])} a {br(reference['p95'])}"
    st.markdown(f"""
| técnica | o que é comparado | ✓ aprovada | ! revisão | ✕ reprovada |
|---|---|---|---|---|
| **GLCM** · contraste, energia | Δ = sintética − real, em IQR dessa métrica entre as CTs reais | \\|Δ\\| ≤ 1 IQR | 1 < \\|Δ\\| ≤ 2 IQR | \\|Δ\\| > 2 IQR |
| **Wavelet** · 2 px, 4 px, entropia | a mesma regra da GLCM | \\|Δ\\| ≤ 1 IQR | 1 < \\|Δ\\| ≤ 2 IQR | \\|Δ\\| > 2 IQR |
| **NEx** · assinatura espectral | NEx da sintética × faixa P5–P95 das CTs reais | entre {nex_band} | fora da faixa | — (NEx não reprova) |
| **Veredito de textura** | a técnica mais severa decide | as três aprovaram | alguma pediu revisão | alguma reprovou |
""")
    st.caption("Em GLCM e Wavelet, a métrica mais severa decide a técnica. NEx inconclusivo (poucas janelas no pulmão) não pesa. O brilho não entra em nenhuma linha desta régua.")


@st.cache_data(show_spinner="Carregando a figura do corte...")
def technique_png(history_dir: str, stamp: float, case_id: str, technique: str) -> bytes:
    board = build_board(Path(history_dir))
    rows = {name: next(row for row in board["rows"][name] if row["case_id"] == case_id) for name in board["names"]}
    return png(technique_figure(Path(history_dir), case_id, technique, rows))


@st.cache_data(show_spinner="Desenhando o mapa regional...")
def map_png(history_dir: str, stamp: float, case_id: str, layer: str, compared: bool) -> bytes:
    return png(map_figure(Path(history_dir), case_id, layer, compared))


@st.cache_data(show_spinner="Carregando as CTs originais...")
def raw_pair_png(history_dir: str, stamp: float, case_id: str, version: str) -> bytes:
    """Mostra os pixels das duas CTs sem mapa, máscara ou transformação."""
    history = load_history(Path(history_dir))
    entry = next(item for item in history["versions"] if item["name"] == version)
    case = load_case(entry, case_id)
    values = np.concatenate((case.original[case.lung_mask], case.synthetic[case.lung_mask]))
    low, high = np.percentile(values, (1, 99))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), squeeze=False)
    for axis, image, title in zip(axes[0], (case.original, case.synthetic), ("CT real · original", "CT sintética · gerada")):
        axis.imshow(image, cmap="gray", vmin=low, vmax=high)
        axis.set_title(title, fontsize=11, fontweight="bold")
        axis.axis("off")
    fig.tight_layout()
    return png(fig)


@st.cache_data(show_spinner=False)
def glcm_tone_counts(history_dir: str, stamp: float, case_id: str, version: str) -> dict[str, tuple[int, float]]:
    """Extremos da discretização oficial, contando somente pixels da ROI."""
    history = load_history(Path(history_dir))
    entry = next(item for item in history["versions"] if item["name"] == version)
    case = load_case(entry, case_id)
    scale = context_from_dict(history["calibration"]).scale
    last_tone = scale.levels - 1

    def counts(image: np.ndarray) -> tuple[tuple[int, float], tuple[int, float]]:
        tones = quantize(image, scale)[case.lung_mask]
        total = len(tones)
        return ((int(np.sum(tones == 0)), 100 * float(np.mean(tones == 0))),
                (int(np.sum(tones == last_tone)), 100 * float(np.mean(tones == last_tone))))

    real_zero, real_last = counts(case.original)
    synthetic_zero, synthetic_last = counts(case.synthetic)
    return {"real_zero": real_zero, "real_last": real_last, "synthetic_zero": synthetic_zero, "synthetic_last": synthetic_last, "last_tone": last_tone}


def texture_chart(rows: list[dict], glcm_ceiling: float | None, wavelet_ceiling: float | None, technique: str):
    """Real × sintética das métricas de uma técnica; cada ponto abre o respectivo par."""
    specs = [
        ("glcm_contrast_real", "glcm_contrast_synthetic", "GLCM · contraste", "contraste"),
        ("glcm_energy_real", "glcm_energy_synthetic", "GLCM · energia", "energia"),
        ("wavelet_energy_real", "wavelet_energy_synthetic", "Wavelet · energia (2 + 4 px)", "energia"),
        ("wavelet_entropy_real", "wavelet_entropy_synthetic", "Wavelet · entropia", "entropia"),
        ("nex_real", "nex_synthetic", "NEx · assinatura espectral", "NEx"),
    ]
    panels = []
    for real_key, synthetic_key, title, label in (spec for spec in specs if spec[2].startswith(technique)):
        data = pd.DataFrame({
            "case_id": [row["case_id"] for row in rows],
            "corte": [case_name(row["case_id"]) for row in rows],
            "real": [number(row, real_key) for row in rows],
            "sintética": [number(row, synthetic_key) for row in rows],
            "veredito": [row.get("wavelet_verdict", "n/a") for row in rows],
        })
        automatic_upper = max(data["real"].max(), data["sintética"].max()) * 1.06
        # Energia GLCM e Wavelet estão em ordens de grandeza diferentes; cada
        # técnica tem sua própria régua comparável. Contraste e entropia ficam
        # sempre na escala dos dados.
        ceiling = glcm_ceiling if title == "GLCM · energia" else wavelet_ceiling if title.startswith("Wavelet · energia") else None
        upper = ceiling if ceiling is not None else automatic_upper
        # entropia normalizada fica toda perto de 0,94: a partir de zero os pontos viram um borrão
        lower = .98 * min(data["real"].min(), data["sintética"].min()) if title in {"Wavelet · entropia", "NEx · assinatura espectral"} else 0
        identity = alt.Chart(pd.DataFrame({"x": [lower, upper], "y": [lower, upper]})).mark_line(color="#8a5a22", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
        selection = alt.selection_point(name="cut", fields=["case_id"], on="click", empty=False)
        points = alt.Chart(data).mark_circle(size=75, color="#0d6b61").encode(
            x=alt.X("real:Q", title=f"real · {label}", scale=alt.Scale(domain=[lower, upper])),
            y=alt.Y("sintética:Q", title=f"sintética · {label}", scale=alt.Scale(domain=[lower, upper])),
            tooltip=[alt.Tooltip("corte:N"), alt.Tooltip("real:Q", format=".4g"), alt.Tooltip("sintética:Q", format=".4g"), alt.Tooltip("veredito:N")],
            size=focused(190, 75),
            stroke=focused("#17243b", "#0a4f48"),
            strokeWidth=focused(1.8, .7),
            opacity=alt.value(.9),
        ).add_params(selection)
        # O título fora do canvas evita o recorte vertical do Vega quando dois
        # gráficos ocupam a mesma linha.
        panels.append((title, (identity + points).properties(height=255).configure_axis(grid=True, gridOpacity=.14).configure_view(stroke=None)))
    return panels


def draw_charts(charts: list, per_row: int = 2) -> None:
    """Distribui os gráficos e devolve o corte escolhido em qualquer painel."""
    clicked_case = None
    for start in range(0, len(charts), per_row):
        for column, item in zip(st.columns(per_row), charts[start:start + per_row]):
            title, chart = item
            column.markdown(f"**{title}**")
            picked = selectable_chart(column, chart, f"texture_cohort_{title}")
            if picked:
                clicked_case = picked
    return clicked_case


FOCUS_CASE = None  # corte em foco; definido depois da barra lateral


def focused(on, off):
    """Destaca o corte em foco. Vem do corte escolhido, não da seleção do gráfico."""
    return alt.condition(alt.datum.case_id == FOCUS_CASE, alt.value(on), alt.value(off))


def selectable_chart(container, chart, key: str) -> str | None:
    """Gráfico clicável quando o Streamlit permite; senão, o mesmo gráfico sem clique.

    A chave inclui o corte em foco: a cada troca de corte (por clique, tabela ou
    barra lateral) os gráficos voltam sem seleção. Sem isso, a seleção antiga de
    um gráfico continuava ativa e puxava o painel de volta para outro par.

    Algumas versões do Streamlit (ex.: 1.50) recusam seleção em gráficos com
    camadas ou compostos; nelas o corte continua sendo aberto pela tabela.
    """
    try:
        event = container.altair_chart(chart, width="stretch", on_select="rerun", selection_mode="cut", key=f"{key}__{FOCUS_CASE}")
    except StreamlitAPIException:
        container.altair_chart(chart, width="stretch")
        return None
    picked = event.selection.get("cut", [])
    case = picked[0].get("case_id") if picked else None
    return case if case != FOCUS_CASE else None


def metric_chart(rows: list[dict], prefix: str, key: str):
    """Como os gráficos do filtro de brilho: diferença em IQR por corte, com as faixas e a cor da classificação."""
    data = pd.DataFrame({
        "case_id": [row["case_id"] for row in rows],
        "corte": [case_name(row["case_id"]) for row in rows],
        "ordem": list(range(1, len(rows) + 1)),
        "z": [number(row, f"{prefix}_{key}_z") for row in rows],
        "% da original": [100 * number(row, f"{prefix}_{key}_synthetic") / number(row, f"{prefix}_{key}_real") for row in rows],
        "classificação": [row[f"{prefix}_{key}_status"] for row in rows],
    })
    limit = max(2.6, float(data["z"].abs().max()) * 1.12)
    scale = alt.Scale(domain=[-limit, limit])
    colors = alt.Scale(domain=list(METRIC_STATUSES), range=["#0b6b32", "#925a10", "#9f1f18"])
    bands = pd.DataFrame([{"início": a, "fim": b, "faixa": f} for a, b, f in ((-limit, -2, "reprovada"), (-2, -1, "revisão"), (-1, 1, "aprovada"), (1, 2, "revisão"), (2, limit, "reprovada"))])
    band_layer = alt.Chart(bands).mark_rect(opacity=.14).encode(
        y=alt.Y("início:Q", scale=scale, title="Δ (IQR das CTs reais)"), y2="fim:Q",
        color=alt.Color("faixa:N", scale=alt.Scale(domain=list(METRIC_STATUSES), range=["#dff3e5", "#fff0c2", "#fde3e1"]), legend=None))
    rules = alt.Chart(pd.DataFrame({"limite": [-2, -1, 1, 2]})).mark_rule(color="#8a5a22", opacity=.6, strokeDash=[4, 4]).encode(y=alt.Y("limite:Q", scale=scale))
    selection = alt.selection_point(name="cut", fields=["case_id"], on="click", empty=False)
    points = alt.Chart(data).mark_circle(filled=True).encode(
        x=alt.X("ordem:Q", title="cortes do dataset"),
        y=alt.Y("z:Q", scale=scale, title="Δ (IQR das CTs reais)"),
        color=alt.Color("classificação:N", scale=colors, legend=None),
        size=focused(285, 118),
        stroke=focused("#17243b", "#24352d"),
        strokeWidth=focused(2.2, .9),
        tooltip=[alt.Tooltip("corte:N"), alt.Tooltip("z:Q", title="diferença (IQR)", format="+.2f"), alt.Tooltip("% da original:Q", format=".0f"), alt.Tooltip("classificação:N")],
    ).add_params(selection)
    # escalas de cor independentes: senão as faixas (tons claros) e os pontos (tons fortes) dividem uma só escala
    return (band_layer + rules + points).resolve_scale(color="independent").properties(height=260).configure_axis(grid=True, gridOpacity=.14).configure_view(stroke=None)


def verdict_matrix(rows: list[dict], mode: str = "textura"):
    """Uma coluna por corte e uma linha por técnica: quem aprovou, quem pediu revisão, quem reprovou."""
    if mode == "brilho":
        combined, techniques, statuses_of, verdict_of = "Brilho (combinado)", BRIGHTNESS_TECHNIQUES, brightness_statuses, brightness_verdict
        why = {"GLCM": glcm_brightness_why, "Wavelet": wavelet_brightness_why}
        reason_of, combined_reason = (lambda row, technique: why[technique](row)), brightness_explanation
    else:
        combined, techniques, statuses_of, verdict_of = "Textura (combinado)", TECHNIQUES, technique_statuses, texture_status
        reason_of, combined_reason = (lambda row, technique: technique_reason(row, technique) or "dentro da faixa"), (lambda row: texture_decision(row)[1])
    order = sorted(rows, key=lambda row: (-METRIC_STATUSES.index(verdict_of(row)), -sum(s != "aprovada" for s in statuses_of(row).values()), row["case_id"]))
    cells = []
    for row in order:
        readings = {combined: (verdict_of(row), combined_reason(row)),
                    **{technique: (status, reason_of(row, technique)) for technique, status in statuses_of(row).items()}}
        for line, (status, reason) in readings.items():
            cells.append({"case_id": row["case_id"], "corte": case_name(row["case_id"]), "técnica": line, "classificação": status, "o que disse": reason})
    selection = alt.selection_point(name="cut", fields=["case_id"], on="click", empty=False)
    return alt.Chart(pd.DataFrame(cells)).mark_rect(stroke="white", strokeWidth=1.5).encode(
        x=alt.X("corte:N", sort=[case_name(row["case_id"]) for row in order], title="cortes (os mais graves à esquerda)", axis=alt.Axis(labelAngle=-60, labelFontSize=9)),
        y=alt.Y("técnica:N", sort=[combined, *techniques], title=None, axis=alt.Axis(labelFontWeight="bold", labelLimit=200)),
        color=alt.Color("classificação:N", scale=STATUS_COLORS, legend=alt.Legend(orient="top", title=None)),
        opacity=focused(1, .82),
        stroke=focused("#17243b", "white"), strokeWidth=focused(2.5, 1.5),
        tooltip=[alt.Tooltip("corte:N"), alt.Tooltip("técnica:N"), alt.Tooltip("classificação:N"), alt.Tooltip("o que disse:N")],
    ).add_params(selection).properties(height=alt.Step(34)).configure_view(stroke=None)


def nex_chart(rows: list[dict], reference: dict):
    """NEx da sintética por corte, com a faixa P5–P95 das CTs reais em verde."""
    data = pd.DataFrame({
        "case_id": [row["case_id"] for row in rows],
        "corte": [case_name(row["case_id"]) for row in rows],
        "ordem": list(range(1, len(rows) + 1)),
        "NEx sintética": [number(row, "nex_synthetic") for row in rows],
        "NEx real": [number(row, "nex_real") for row in rows],
        "classificação": [NEX_STATUS[row["nex_status"]] for row in rows],
    })
    values = [*data["NEx sintética"], reference["p5"], reference["p95"]]
    pad = .08 * (max(values) - min(values))
    scale = alt.Scale(domain=[min(values) - pad, max(values) + pad])
    band = alt.Chart(pd.DataFrame({"início": [reference["p5"]], "fim": [reference["p95"]]})).mark_rect(opacity=.14, color="#2f9e57").encode(y=alt.Y("início:Q", scale=scale, title="NEx da sintética"), y2="fim:Q")
    selection = alt.selection_point(name="cut", fields=["case_id"], on="click", empty=False)
    points = alt.Chart(data).mark_circle(filled=True).encode(
        x=alt.X("ordem:Q", title="cortes do dataset"),
        y=alt.Y("NEx sintética:Q", scale=scale, title="NEx da sintética"),
        color=alt.Color("classificação:N", scale=STATUS_COLORS, legend=None),
        size=focused(285, 118),
        stroke=focused("#17243b", "#24352d"),
        strokeWidth=focused(2.2, .9),
        tooltip=[alt.Tooltip("corte:N"), alt.Tooltip("NEx sintética:Q", format=".3f"), alt.Tooltip("NEx real:Q", format=".3f"), alt.Tooltip("classificação:N")],
    ).add_params(selection)
    return (band + points).properties(height=260).configure_axis(grid=True, gridOpacity=.14).configure_view(stroke=None)


def brightness_chart(rows: list[dict]):
    """Os gráficos oficiais de brilho, com faixas e seleção do corte por clique."""
    data = pd.DataFrame({
        "case_id": [row["case_id"] for row in rows],
        "corte": [case_name(row["case_id"]) for row in rows],
        "ordem": list(range(1, len(rows) + 1)),
        "delta_mu": [number(row, "delta_mu") for row in rows],
        "sat31": [number(row, "sat31_pct") for row in rows],
        "decisão": [STATUS_TEXT[row["brightness_status"]] for row in rows],
    })
    selection = alt.selection_point(name="cut", fields=["case_id"], on="click", empty=False)
    decision_colors = alt.Scale(domain=["aprovada", "revisão manual", "rejeição direta"], range=["#0b6b32", "#925a10", "#9f1f18"])
    base = alt.Chart(data).mark_circle(filled=True).encode(
        x=alt.X("ordem:Q", title="cortes do dataset"),
        color=alt.Color("decisão:N", title="GLCM", scale=decision_colors),
        size=focused(285, 118),
        stroke=focused("#17243b", "#24352d"),
        strokeWidth=focused(2.2, .9),
        tooltip=[alt.Tooltip("corte:N"), alt.Tooltip("delta_mu:Q", title="Δμ", format="+.3f"), alt.Tooltip("sat31:Q", title="Sat31 (%)", format=".2f"), alt.Tooltip("decisão:N")],
        opacity=alt.value(1),
    ).add_params(selection)
    delta_limit = max(.70, float(data["delta_mu"].abs().max()) * 1.12)
    sat_limit = max(16.0, float(data["sat31"].max()) * 1.12)
    delta_scale, sat_scale = alt.Scale(domain=[-delta_limit, delta_limit]), alt.Scale(domain=[0, sat_limit])
    band_colors = alt.Scale(domain=["aprovada", "revisão manual", "rejeição direta"], range=["#dff3e5", "#fff0c2", "#fde3e1"])
    delta_bands = pd.DataFrame([{"início": -delta_limit, "fim": -OFFICIAL_REJECT, "faixa": "rejeição direta"}, {"início": -OFFICIAL_REJECT, "fim": -OFFICIAL_REVIEW, "faixa": "revisão manual"}, {"início": -OFFICIAL_REVIEW, "fim": OFFICIAL_REVIEW, "faixa": "aprovada"}, {"início": OFFICIAL_REVIEW, "fim": OFFICIAL_REJECT, "faixa": "revisão manual"}, {"início": OFFICIAL_REJECT, "fim": delta_limit, "faixa": "rejeição direta"}])
    sat_bands = pd.DataFrame([{"início": 0, "fim": OFFICIAL_SATURATION, "faixa": "aprovada"}, {"início": OFFICIAL_SATURATION, "fim": sat_limit, "faixa": "rejeição direta"}])
    def bands(frame: pd.DataFrame, scale: alt.Scale, title: str):
        return alt.Chart(frame).mark_rect(opacity=.14).encode(y=alt.Y("início:Q", scale=scale, title=title), y2="fim:Q", color=alt.Color("faixa:N", scale=band_colors, legend=None))
    review = alt.Chart(pd.DataFrame({"limite": [-OFFICIAL_REVIEW, OFFICIAL_REVIEW]})).mark_rule(color="#b7791f", opacity=.7, strokeDash=[4, 4]).encode(y=alt.Y("limite:Q", scale=delta_scale, title="Δμ · sintética − real"))
    reject = alt.Chart(pd.DataFrame({"limite": [-OFFICIAL_REJECT, OFFICIAL_REJECT]})).mark_rule(color="#b42318", opacity=.7, strokeDash=[4, 4]).encode(y=alt.Y("limite:Q", scale=delta_scale, title="Δμ · sintética − real"))
    saturation_rule = alt.Chart(pd.DataFrame({"limite": [OFFICIAL_SATURATION]})).mark_rule(color="#b42318", opacity=.7, strokeDash=[4, 4]).encode(y=alt.Y("limite:Q", scale=sat_scale, title="Sat31 (%)"))
    delta = (bands(delta_bands, delta_scale, "Δμ · sintética − real") + review + reject + base.encode(y=alt.Y("delta_mu:Q", title="Δμ · sintética − real", scale=delta_scale))).properties(title="Δμ por corte", width=390, height=280)
    saturation = (bands(sat_bands, sat_scale, "Sat31 (%)") + saturation_rule + base.encode(y=alt.Y("sat31:Q", title="Sat31 (%)", scale=sat_scale), color=alt.Color("decisão:N", title="GLCM", scale=decision_colors, legend=None))).properties(title="Saturação no tom 31", width=390, height=280)
    return alt.hconcat(delta, saturation).configure_axis(grid=True, gridOpacity=.14).configure_view(stroke=None)


def metrics_table(rows: list[dict], mode: str) -> pd.DataFrame:
    if mode == "brilho":
        data = [{"case_id": row["case_id"], "corte": case_name(row["case_id"]), "decisão de brilho": brightness_decision(row), **brightness_statuses(row),
                 "por quê": brightness_explanation(row), "|Δμ|": number(row, "abs_delta_mu"), "Sat31 (%)": number(row, "sat31_pct"),
                 "Wavelet LL": metric_reading(row, "wavelet", "ll"), "densidade (não decide)": density_reading(row)} for row in rows]
    else:
        data = [{"case_id": row["case_id"], "corte": case_name(row["case_id"]), "leitura de textura": texture_decision(row)[0],
                 **technique_statuses(row), "por quê": texture_decision(row)[1],
                 "GLCM contraste": metric_reading(row, "glcm", "contrast"), "GLCM energia": metric_reading(row, "glcm", "energy"),
                 "Wavelet 2 px": wavelet_metric_reading(row, "fine"), "Wavelet 4 px": wavelet_metric_reading(row, "coarse"), "Wavelet entropia": wavelet_metric_reading(row, "entropy"),
                 "tipo (Wavelet)": row.get("wavelet_verdict", "n/a"), "NEx leitura": nex_reading(row),
                 "GLCM contraste (%)": number(row, "glcm_contrast_pct"), "Wavelet 2 px (% da original)": number(row, "wavelet_fine_pct"), "Wavelet 4 px (% da original)": number(row, "wavelet_coarse_pct"), "NEx sintética": number(row, "nex_synthetic")} for row in rows]
    frame = pd.DataFrame(data)
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].round(3)
    return frame


def show_method_cards(mode: str, reference: dict):
    if mode == "brilho":
        st.markdown("#### Limites do brilho")
        st.write("**Nesta aba, só o brilho decide; a textura não entra.** GLCM e Wavelet classificam o par cada uma sozinha, e o **veredito de brilho** é o da mais severa.")
        st.markdown(f"""
| técnica | o que é comparado | ✓ aprovada | ! revisão manual | ✕ rejeição direta |
|---|---|---|---|---|
| **GLCM** · \\|Δμ\\| | diferença do tom médio do pulmão, na régua de 32 tons | ≤ {br(OFFICIAL_REVIEW)} | {br(OFFICIAL_REVIEW)} a {br(OFFICIAL_REJECT)} | > {br(OFFICIAL_REJECT)} |
| **GLCM** · Sat31 | % de pixels do pulmão no tom mais claro (31) | ≤ 15% | — | > 15% |
| **Wavelet** · LL | Δ = LL da sintética − LL da real, em IQR da LL entre as CTs reais | \\|Δ\\| ≤ 1 IQR | 1 < \\|Δ\\| ≤ 2 IQR | \\|Δ\\| > 2 IQR |
| **Veredito de brilho** | a técnica mais severa decide | as duas aprovaram | alguma pediu revisão | alguma reprovou |
""")
        st.caption("Na GLCM, o pior entre |Δμ| e Sat31 decide a técnica. Os limites de |Δμ| são 1 e 2 IQR das médias pulmonares das CTs reais. A densidade aparece só como contexto e não decide.")
        with st.expander("O que cada técnica mede", expanded=False):
            st.dataframe(pd.DataFrame([
                ["GLCM · |Δμ|", "quanto o tom médio da sintética se afastou do da real, na mesma régua de 32 tons usada pela GLCM"],
                ["GLCM · Sat31", "quantos pixels estouraram no tom mais claro da régua; pega imagens lavadas mesmo com Δμ pequeno"],
                ["Wavelet · LL", "média local de 4×4 px (banda LL): acompanha o brilho e ignora o detalhe; pega cortes claros que o Δμ deixa passar"],
                ["Wavelet · energia e entropia", "não respondem ao brilho (só ao detalhe), por isso ficam na aba de textura"],
                ["Densidade (contexto)", "pixels fora da faixa P1–P99 real e deslocamento da mediana; não decide"],
            ], columns=["técnica", "o que ela mede"]), hide_index=True, width="stretch")
    else:
        st.markdown("#### Limites da textura")
        st.write("**Nesta aba, só a textura decide; o brilho não entra.** Cada técnica classifica o par sozinha, e o **veredito de textura** é o da mais severa.")
        show_texture_limits(reference)
        with st.expander("O que cada técnica mede", expanded=False):
            st.dataframe(pd.DataFrame([
                ["GLCM · contraste", "quanto o tom muda entre pixels vizinhos; queda sugere suavização/perda de alta frequência"],
                ["GLCM · energia", "uniformidade dos pares de tons; pode subir quando a imagem fica mais lisa"],
                ["Wavelet · 2 px e 4 px", "quantidade de detalhe na escala fina (~2 px) e na grossa (~4 px)"],
                ["Wavelet · entropia", "variação/complexidade da distribuição de detalhes"],
                ["Wavelet · tipo", "rótulo descritivo (borrou, ruído, perdeu contraste…); descreve a mudança, não decide"],
                ["NEx · assinatura espectral", "excesso ou falta de estrutura perto do limite de resolução; compara a sintética com a faixa P5–P95 das CTs reais"],
            ], columns=["técnica", "o que ela mede"]), hide_index=True, width="stretch")


def show_case(board: dict, history_dir: Path, stamp: float, case_id: str, version: str, mode: str):
    row = next(item for item in board["rows"][version] if item["case_id"] == case_id)
    st.markdown("---")
    title, close = st.columns([5, 1], vertical_alignment="bottom")
    title.subheader(f"Par em foco · {case_name(case_id)}")
    if close.button("Fechar par", key=f"close_{case_id}", width="stretch"):
        st.session_state["pending_case"] = None
        st.session_state["pending_mode"] = mode
        st.rerun()
    if mode == "brilho":
        verdict, statuses = brightness_verdict(row), brightness_statuses(row)
        show, icon = STATUS_BOX[verdict]
        show(f"#### Veredito de brilho: {BRIGHTNESS_LABEL[verdict]}\n{verdict_summary(statuses, verdict)}", icon=icon)
        why = {"GLCM": glcm_brightness_why, "Wavelet": wavelet_brightness_why}
        details = {"GLCM": f"{STATUS_MARK[statuses['GLCM']]} **|Δμ|**: {br(number(row, 'abs_delta_mu'))} · **Sat31**: {number(row, 'sat31_pct'):.1f}%".replace(".", ","),
                   "Wavelet": f"{STATUS_MARK[statuses['Wavelet']]} **LL**: real {br(number(row, 'wavelet_ll_real'))} → sintética {br(number(row, 'wavelet_ll_synthetic'))} (0 = P1, 1 = P99)"}
        for column, technique in zip(st.columns(2), BRIGHTNESS_TECHNIQUES):
            with column:
                show, icon = STATUS_BOX[statuses[technique]]
                show(f"**{technique}: {BRIGHTNESS_LABEL[statuses[technique]]}**  \n{why[technique](row)}", icon=icon)
                st.markdown(details[technique])
        st.caption(f"Contexto, não decide · densidade: {density_reading(row)}")
        choices = {"CT real × sintética (Δμ)": "delta_mu", "Densidade · pixels fora da faixa real": "densidade", "Brilho local (Wavelet LL)": "ll"}
    else:
        reference = board["history"]["calibration"]["nex_reference"]
        verdict = texture_status(row)
        show, icon = STATUS_BOX[verdict]
        show(f"#### Veredito de textura: {TEXTURE_LABEL[verdict]}\n{verdict_summary(technique_statuses(row), verdict)}", icon=icon)
        statuses = technique_statuses(row)
        for column, technique in zip(st.columns(3), TECHNIQUES):
            with column:
                show, icon = STATUS_BOX[statuses[technique]]
                show(f"**{technique}: {statuses[technique]}**  \n{technique_why(row, technique, reference)}", icon=icon)
                if technique == "NEx":
                    st.markdown(f"{STATUS_MARK[statuses['NEx']]} **NEx**: sintética {br(number(row, 'nex_synthetic'))} · real {br(number(row, 'nex_real'))}")
                else:
                    prefix, metrics = TECHNIQUE_METRICS[technique]
                    st.markdown("  \n".join(metric_line(row, prefix, key, label.removeprefix(technique + " ")) for key, label in metrics.items()))
                if technique == "Wavelet":
                    st.caption(f"tipo de mudança (descreve, não decide): {row.get('wavelet_verdict', 'n/a')}")
        choices = {"GLCM · contraste": "glcm_contraste", "Wavelet · detalhe fino": "wavelet_fina", "Wavelet · detalhe grosso": "wavelet_grossa"}
    st.image(raw_pair_png(str(history_dir), stamp, case_id, version), width="stretch")
    st.caption("CTs brutas lado a lado — sem mapa, máscara ou métrica sobreposta. A mesma janela de visualização é usada nas duas imagens para permitir comparação direta.")
    tones = glcm_tone_counts(str(history_dir), stamp, case_id, version)
    # na textura os extremos são detalhe técnico; no brilho eles explicam o Sat31
    with st.expander("Extremos da discretização GLCM", expanded=mode == "brilho"):
        metrics = st.columns(4)
        for column, label, key in zip(metrics, ("CT real · tom 0", f"CT real · tom {tones['last_tone']}", "CT sintética · tom 0", f"CT sintética · tom {tones['last_tone']}"), ("real_zero", "real_last", "synthetic_zero", "synthetic_last")):
            _, pct = tones[key]
            column.metric(label, f"{pct:.2f}%", "dos pixels da ROI")
        st.caption(f"A régua oficial discretiza a ROI pulmonar em {tones['last_tone'] + 1} tons. Os cartões mostram a porcentagem no menor tom (0) e no maior tom ({tones['last_tone']}); não contam pares de vizinhos da GLCM.")
    with st.expander("Entender a métrica neste par", expanded=False):
        st.caption("Esta visualização mostra o que a técnica calcula, não a CT bruta acima.")
        technique = st.segmented_control("Técnica", list(choices), default=list(choices)[0], key=f"technique_{mode}")
        st.image(technique_png(str(history_dir), stamp, case_id, choices[technique]), width="stretch")
        st.caption(TECHNIQUE_FIGURES[choices[technique]])
    with st.expander("Mapa regional — onde a diferença aparece", expanded=False):
        st.caption("Use este mapa apenas para localizar um padrão já visto no par. Ele divide o pulmão em blocos de 16 px; não é uma nova regra de aprovação.")
        columns = st.columns(2)
        default_mode = "Comparado com a original" if mode == "textura" else "Valor de cada imagem"
        map_mode = columns[0].segmented_control("Mostrar", ["Valor de cada imagem", "Comparado com a original"], default=default_mode, key=f"map_mode_{mode}")
        compared = map_mode == "Comparado com a original"
        layers = {label: key for key, label in MAP_LAYERS.items() if compared or key != "veredito"}
        layer = columns[1].segmented_control("Camada", list(layers), default=list(layers)[0], key=f"map_layer_{mode}_{compared}")
        st.image(map_png(str(history_dir), stamp, case_id, layers[layer], compared), width="stretch")
        if compared:
            st.caption("Comparação bloco a bloco com a original. Só é válida quando a anatomia é a mesma; em Dgen, prefira “Valor de cada imagem”.")


st.set_page_config(page_title="Avaliação de CT sintética", page_icon="◌", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1450px; padding-top: 2.2rem; padding-bottom: 4rem;}
[data-testid="stMetric"] {background: #f5f7f6; border: 1px solid #e2e8e5; border-radius: 10px; padding: .85rem 1rem;}
h1 {letter-spacing: -.04em;} h2, h3 {letter-spacing: -.025em;}
[data-testid="stSidebar"] {border-right: 1px solid #e5e9e7;}
</style>""", unsafe_allow_html=True)

st.title("Avaliação de CT pulmonar sintética")
st.caption("Escolha o dataset, veja a coorte e então abra um par. As métricas não substituem revisão radiológica.")
with st.sidebar:
    st.header("1 · Dataset")
    dataset = st.radio("Dataset em análise", list(DATASETS), format_func=lambda item: f"{item} · {'mesma anatomia' if item == 'Dsr' else 'só máscara'}")
    config = DATASETS[dataset]
    history_dir = resolve(config["history"])
    st.caption(config["description"])
    st.caption(f"Histórico: `{short(history_dir)}`")

history = load_history(history_dir)
if history is None:
    st.info(f"{dataset} ainda não foi avaliado neste painel. A configuração usa os arquivos reais de `{config['folder']}` e cria um histórico próprio, sem misturar os datasets.")
    if st.button(f"Avaliar {dataset} e abrir painel", type="primary"):
        try:
            with st.spinner(f"Avaliando {dataset}..."):
                message = add_version(history_dir, dataset, resolve(config["folder"]), resolve(config["masks"]))
                write_outputs(history_dir, build_board(history_dir))
            st.success(message)
            st.rerun()
        except PlacarError as error:
            st.error(str(error))
    st.stop()

if any(key not in history["calibration"] for key in ("nex_reference", "wavelet_iqr", "glcm_iqr", "ll_iqr")):
    st.warning("Este histórico foi calculado antes da integração do NEx ou da classificação por métrica GLCM/Wavelet. Recalcule as versões para aplicar as faixas reais aos resultados.")
    if st.button("Recalcular versões", type="primary"):
        with st.spinner("Recalculando GLCM, Wavelet e NEx..."):
            st.success(refresh_history(history_dir))
        st.rerun()
    st.stop()

try:
    board = build_board(history_dir)
except PlacarError as error:
    st.error(str(error))
    st.stop()
if board is None:
    st.error("O histórico não tem versões avaliadas.")
    st.stop()
calibration_limits(history)
names, version = board["names"], board["names"][-1]
rows = board["rows"][version]
stamp = (history_dir / "historico.json").stat().st_mtime

# Um clique num ponto do gráfico chega no final de uma execução. Guardamos o
# corte para aplicá-lo antes que os widgets da barra lateral sejam montados.
if "pending_case" in st.session_state:
    st.session_state[f"case_filter_{st.session_state.pop('pending_mode', 'textura')}"] = "todos"
    st.session_state["case_selector"] = st.session_state.pop("pending_case")

with st.sidebar:
    st.markdown("---")
    st.header("2 · Corte")
    active_mode = st.radio("Filtro a explorar", ["Análise de textura", "Filtro de brilho"], index=0)
    mode = "textura" if active_mode == "Análise de textura" else "brilho"
    if mode == "brilho":
        selection = st.selectbox("Mostrar cortes", ["todos", *BRIGHTNESS_LABEL.values(), *(f"{technique} fora da faixa" for technique in BRIGHTNESS_TECHNIQUES)], key="case_filter_brilho")
        choices = [row["case_id"] for row in rows if selection == "todos" or brightness_decision(row) == selection
                   or selection.endswith("fora da faixa") and brightness_statuses(row)[selection.split()[0]] in {"revisão", "reprovada"}]
    else:
        selection = st.selectbox("Mostrar cortes", ["todos", *TEXTURE_LABEL.values(), *(f"{technique} fora da faixa" for technique in TECHNIQUES)], key="case_filter_textura")
        choices = [row["case_id"] for row in rows if selection == "todos" or texture_decision(row)[0] == selection
                   or selection.endswith("fora da faixa") and technique_statuses(row)[selection.split()[0]] in {"revisão", "reprovada"}]
    case_id = st.selectbox("Corte para inspecionar", choices, index=None, placeholder="nenhum par aberto", format_func=case_name, key="case_selector")
FOCUS_CASE = case_id

show_method_cards(mode, history["calibration"]["nex_reference"])
st.markdown(f"#### Resumo do dataset · {dataset}")
if mode == "brilho":
    statuses = pd.Series([brightness_decision(row) for row in rows]).value_counts()
    first, second, third = st.columns(3)
    first.metric("✓ Brilho aprovado", int(statuses.get("aprovada", 0)), f"de {len(rows)} cortes", delta_color="off", delta_arrow="off")
    second.metric("! Revisão manual", int(statuses.get("revisão manual", 0)), "alguma técnica pediu revisão", delta_color="off", delta_arrow="off")
    third.metric("✕ Rejeição direta", int(statuses.get("rejeição direta", 0)), "alguma técnica reprovou", delta_color="off", delta_arrow="off")
    st.caption("A textura não entra nesta contagem; ela é avaliada só na “Análise de textura”.")
    st.markdown("##### Quem apontou o quê")
    for column, technique in zip(st.columns(2), BRIGHTNESS_TECHNIQUES):
        counts = pd.Series([brightness_statuses(row)[technique] for row in rows]).value_counts()
        column.metric(f"{technique}: fora da faixa", int(counts.get("revisão", 0) + counts.get("reprovada", 0)),
                      " · ".join(f"{int(counts.get(status, 0))} {BRIGHTNESS_LABEL[status]}" for status in METRIC_STATUSES if counts.get(status, 0)), delta_color="off", delta_arrow="off")
    only_wavelet = sum(statuses_["GLCM"] == "aprovada" and statuses_["Wavelet"] != "aprovada" for statuses_ in map(brightness_statuses, rows))
    if only_wavelet:
        st.caption(f"{only_wavelet} corte(s) passariam só com a GLCM e foram apontados pela Wavelet LL.")
else:
    statuses = pd.Series([texture_decision(row)[0] for row in rows]).value_counts()
    first, second, third = st.columns(3)
    first.metric("✓ Textura preservada", int(statuses.get("textura preservada", 0)), f"de {len(rows)} cortes", delta_color="off", delta_arrow="off")
    second.metric("! Revisão necessária", int(statuses.get("revisão necessária", 0)), "alguma técnica pediu revisão", delta_color="off", delta_arrow="off")
    third.metric("✕ Reprovadas", int(statuses.get("reprovada", 0)), "alguma técnica reprovou", delta_color="off", delta_arrow="off")
    st.caption("O brilho não entra nesta contagem; ele é avaliado só no “Filtro de brilho”.")
    st.markdown("##### Quem apontou o quê")
    for column, technique in zip(st.columns(3), TECHNIQUES):
        counts = pd.Series([technique_statuses(row)[technique] for row in rows]).value_counts()
        column.metric(f"{technique}: fora da faixa", int(counts.get("revisão", 0) + counts.get("reprovada", 0)),
                      " · ".join(f"{int(counts.get(status, 0))} {status}" for status in (*METRIC_STATUSES, "inconclusivo") if counts.get(status, 0)).replace("inconclusivo", "inconcl."), delta_color="off", delta_arrow="off")

if not choices:
    st.info("Nenhum corte nessa categoria. Escolha outro filtro em “Mostrar cortes”.")
elif case_id is None:
    st.info("Nenhum par aberto. Clique num ponto, célula ou linha em “Todos os cortes do dataset”, ou escolha um corte na barra lateral.")
else:
    show_case(board, history_dir, stamp, case_id, version, mode)
st.markdown("---")
st.subheader("Todos os cortes do dataset")
st.caption("Clique em qualquer ponto, célula ou linha para abrir aquele par no topo, em “Par em foco”.")
if mode == "textura":
    st.markdown("#### Veredito por corte e por técnica")
    st.caption("Cada coluna é um corte. A primeira linha é o veredito de textura; as outras mostram o que cada técnica disse. Passe o mouse para ver o motivo e clique para abrir o corte.")
    clicked_case = selectable_chart(st, verdict_matrix(rows), "verdict_matrix")
    st.markdown("#### Detalhe de cada técnica")
    glcm_tab, wavelet_tab, nex_tab = st.tabs(["GLCM", "Wavelet", "NEx"])
    with glcm_tab:
        st.caption("Cada ponto é um corte, na diferença sintética − real medida em IQR das CTs reais. Verde = aprovada, amarelo = revisão, vermelho = reprovada.")
        for column, (key, label) in zip(st.columns(len(GLCM_METRICS)), GLCM_METRICS.items()):
            counts = pd.Series([row[f"glcm_{key}_status"] for row in rows]).value_counts()
            column.markdown(f"**{label}**")
            column.caption(" · ".join(f"{int(counts.get(status, 0))} {status}" for status in METRIC_STATUSES))
            clicked_case = selectable_chart(column, metric_chart(rows, "glcm", key), f"glcm_metric_{key}") or clicked_case
        st.markdown("##### Real × sintética")
        glcm_choice = st.segmented_control("Escala · energia GLCM", ["Automática", "Até 0,10", "Até 0,20"], default="Automática", help="Use a mesma régua ao alternar Dsr e Dgen.") or "Automática"
        glcm_ceiling = {"Automática": None, "Até 0,10": .10, "Até 0,20": .20}[glcm_choice]
        st.caption("A linha tracejada é a identidade: real = sintética.")
        clicked_case = draw_charts(texture_chart(rows, glcm_ceiling, None, "GLCM")) or clicked_case
    with wavelet_tab:
        st.caption("Cada ponto é um corte, na diferença sintética − real medida em IQR das CTs reais. Verde = aprovada, amarelo = revisão, vermelho = reprovada.")
        for column, (key, label) in zip(st.columns(len(WAVELET_METRICS)), WAVELET_METRICS.items()):
            counts = pd.Series([row[f"wavelet_{key}_status"] for row in rows]).value_counts()
            column.markdown(f"**{label}**")
            column.caption(" · ".join(f"{int(counts.get(status, 0))} {status}" for status in METRIC_STATUSES))
            clicked_case = selectable_chart(column, metric_chart(rows, "wavelet", key), f"wavelet_metric_{key}") or clicked_case
        kinds = pd.Series([row.get("wavelet_verdict", "n/a") for row in rows]).value_counts()
        st.caption("Tipo de mudança (descreve, não decide): " + " · ".join(f"{int(count)} {kind}" for kind, count in kinds.items()))
        st.markdown("##### Real × sintética")
        wavelet_choice = st.segmented_control("Escala · energia Wavelet", ["Automática", "Até 0,01", "Até 0,02", "Até 0,05"], default="Automática", help="Wavelet usa valores muito menores do que energia GLCM.") or "Automática"
        wavelet_ceiling = {"Automática": None, "Até 0,01": .01, "Até 0,02": .02, "Até 0,05": .05}[wavelet_choice]
        st.caption("A linha tracejada é a identidade: real = sintética.")
        clicked_case = draw_charts(texture_chart(rows, None, wavelet_ceiling, "Wavelet")) or clicked_case
    with nex_tab:
        reference = history["calibration"]["nex_reference"]
        counts = pd.Series([NEX_STATUS[row["nex_status"]] for row in rows]).value_counts()
        st.caption(f"NEx da sintética por corte. A faixa verde é P5–P95 das CTs reais ({br(reference['p5'])} a {br(reference['p95'])}); fora dela, revisão. Cinza = inconclusivo (poucas janelas na ROI). "
                   + " · ".join(f"{int(counts.get(status, 0))} {status}" for status in (*METRIC_STATUSES, "inconclusivo") if counts.get(status, 0)))
        clicked_case = selectable_chart(st, nex_chart(rows, reference), "nex_metric") or clicked_case
        st.markdown("##### Real × sintética")
        clicked_case = draw_charts(texture_chart(rows, None, None, "NEx"), per_row=2) or clicked_case
    if clicked_case and clicked_case != case_id:
        st.session_state["pending_case"] = clicked_case
        st.session_state["pending_mode"] = "textura"
        st.rerun()
else:
    st.markdown("#### Veredito por corte e por técnica")
    st.caption("Cada coluna é um corte. A primeira linha é o veredito de brilho; as outras mostram o que cada técnica disse. Passe o mouse para ver o motivo e clique para abrir o corte.")
    clicked_case = selectable_chart(st, verdict_matrix(rows, "brilho"), "brightness_matrix")
    st.markdown("#### Detalhe de cada técnica")
    glcm_tab, wavelet_tab = st.tabs(["GLCM · Δμ e Sat31", "Wavelet · LL"])
    with glcm_tab:
        st.caption("Cores pela classificação da GLCM: verde = aprovada, amarelo = revisão manual, vermelho = rejeição direta. No Sat31 não há faixa amarela: acima de 15% já reprova.")
        clicked_case = selectable_chart(st, brightness_chart(rows), "brightness_cohort") or clicked_case
    with wavelet_tab:
        agree = sum(len(set(brightness_statuses(row).values())) == 1 for row in rows)
        st.caption(f"Δ = LL da sintética − LL da real, em IQR da LL entre as CTs reais. Verde = aprovada, amarelo = revisão, vermelho = reprovada. Concorda com a GLCM em {agree} de {len(rows)} cortes.")
        clicked_case = selectable_chart(st, metric_chart(rows, "wavelet", "ll"), "wavelet_ll") or clicked_case
    if clicked_case and clicked_case != case_id:
        st.session_state["pending_case"] = clicked_case
        st.session_state["pending_mode"] = "brilho"
        st.rerun()
with st.expander("Ver todos os cortes e o motivo", expanded=True):
    table = metrics_table(rows, mode)
    # reset_index: a linha clicada volta como posição, e ela precisa bater com a ordem mostrada
    # mais graves primeiro; dentro de cada grupo, o maior desvio
    severity = {label: index for labels in (BRIGHTNESS_LABEL, TEXTURE_LABEL) for index, label in enumerate(reversed(labels.values()))}
    verdict_column, spread_column = ("decisão de brilho", "|Δμ|") if mode == "brilho" else ("leitura de textura", "NEx sintética")
    table = table.assign(_ordem=table[verdict_column].map(severity)).sort_values(["_ordem", spread_column], ascending=[True, False]).drop(columns="_ordem").reset_index(drop=True)
    table.insert(0, "em foco", np.where(table["case_id"] == case_id, "●", ""))
    st.caption("Clique numa linha para abrir o corte em “Par em foco”.")
    event = st.dataframe(table, hide_index=True, width="stretch", height=320, column_order=[column for column in table.columns if column != "case_id"],
                         on_select="rerun", selection_mode="single-row", key=f"cases_{mode}__{case_id}")
    selected = event.selection.rows
    clicked = table.iloc[selected[0]]["case_id"] if selected else None
    # a chave muda com o corte em foco, então toda seleção aqui é um clique novo
    if clicked and clicked != case_id:
        st.session_state["pending_case"] = clicked
        st.session_state["pending_mode"] = mode
        st.rerun()
    if mode == "brilho":
        st.caption("A tabela ignora completamente a textura. As colunas GLCM e Wavelet mostram a classificação de cada técnica; “decisão de brilho” é a mais severa. A densidade aparece só como contexto.")
    else:
        st.caption("A tabela ignora completamente o brilho. As colunas GLCM, Wavelet e NEx mostram a classificação de cada técnica; “leitura de textura” é a pior delas.")
    st.caption("Os pontos e linhas resumem cortes do dataset; fases do mesmo paciente não devem ser interpretadas como pacientes independentes.")

with st.expander("Sobre esta avaliação e as versões", expanded=False):
    st.caption("Informação de rastreabilidade: de onde vieram os cortes e como as versões são comparadas. Não é necessária para inspecionar um par.")
    st.caption(f"{len(names)} versão(ões) · {board['n_cases']} cortes comuns · referência da régua: {board['history']['reference']}")
    st.dataframe(pd.DataFrame([{"versão": item["name"], "modo": item.get("mode", "?"), "cortes válidos": item["n_cases"], "pasta": short(item["folder"])} for item in board["history"]["versions"]]), hide_index=True, width="stretch")
    if len(names) > 1:
        st.markdown("#### Comparação entre versões")
        header, data = technique_table(board, mode)
        st.dataframe(pd.DataFrame(data, columns=header), hide_index=True, width="stretch")
    st.markdown(f"O pareamento exige a anatomia correspondente; no modo mesma anatomia, correlação abaixo de {PAIR_MIN_CORRELATION} é sinal de arquivo suspeito. Dgen usa o modo `{MASK_ONLY}` para evitar comparação espacial indevida.")
    target = st.selectbox("Remover versão do histórico", names, key="remove_version")
    if st.button("Remover versão selecionada", key="remove"):
        remove_version(history_dir, target)
        st.rerun()
