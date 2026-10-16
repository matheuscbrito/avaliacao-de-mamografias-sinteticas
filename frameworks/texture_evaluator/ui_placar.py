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

from frameworks.texture_evaluator.placar import PlacarError, add_version, build_board, load_case, load_history, remove_version, technique_table, write_outputs
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


def official_brightness_status(row: dict) -> str:
    """Gatekeeper oficial (somente Δμ e saturação no tom 31), como o framework já decidiu em `run_mvp`."""
    return STATUS_TEXT[row["brightness_status"]]


def brightness_explanation(row: dict) -> str:
    delta, saturation = number(row, "abs_delta_mu"), number(row, "sat31_pct")
    if saturation > OFFICIAL_SATURATION:
        return f"Sat31 {saturation:.1f}% ultrapassa 15%"
    if delta > OFFICIAL_REJECT:
        return f"|Δμ| {delta:.3f} ultrapassa {br(OFFICIAL_REJECT)}"
    if delta > OFFICIAL_REVIEW:
        return f"|Δμ| {delta:.3f} está entre {br(OFFICIAL_REVIEW)} e {br(OFFICIAL_REJECT)}"
    return f"|Δμ| {delta:.3f} e Sat31 {saturation:.1f}% dentro do gatekeeper"


def texture_explanation(row: dict) -> str:
    energy = 100 * number(row, "glcm_energy_synthetic") / number(row, "glcm_energy_real")
    text = f"GLCM contraste {number(row, 'glcm_contrast_pct'):.0f}%; energia GLCM {energy:.0f}%; Wavelet 2 px {number(row, 'wavelet_fine_pct'):.0f}%; Wavelet 4 px {number(row, 'wavelet_coarse_pct'):.0f}%"
    verdict = row.get("wavelet_verdict", "")
    return text + (f" · leitura exploratória: {verdict}" if verdict and verdict != "textura preservada" else "")


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


def texture_chart(rows: list[dict], glcm_ceiling: float | None, wavelet_ceiling: float | None):
    """Real × sintética por técnica; passar o mouse num ponto mostra o corte."""
    specs = [
        ("glcm_contrast_real", "glcm_contrast_synthetic", "GLCM · contraste", "contraste"),
        ("glcm_energy_real", "glcm_energy_synthetic", "GLCM · energia", "energia"),
        ("wavelet_energy_real", "wavelet_energy_synthetic", "Wavelet · energia (2 + 4 px)", "energia"),
        ("wavelet_entropy_real", "wavelet_entropy_synthetic", "Wavelet · entropia", "entropia"),
    ]
    panels = []
    for real_key, synthetic_key, title, label in specs:
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
        lower = .98 * min(data["real"].min(), data["sintética"].min()) if title == "Wavelet · entropia" else 0
        identity = alt.Chart(pd.DataFrame({"x": [lower, upper], "y": [lower, upper]})).mark_line(color="#8a5a22", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
        points = alt.Chart(data).mark_circle(size=75, color="#0d6b61").encode(
            x=alt.X("real:Q", title=f"real · {label}", scale=alt.Scale(domain=[lower, upper])),
            y=alt.Y("sintética:Q", title=f"sintética · {label}", scale=alt.Scale(domain=[lower, upper])),
            tooltip=[alt.Tooltip("corte:N"), alt.Tooltip("real:Q", format=".4g"), alt.Tooltip("sintética:Q", format=".4g"), alt.Tooltip("veredito:N")],
            opacity=alt.value(.8),
        )
        panels.append((identity + points).properties(title=title, height=255).configure_axis(grid=True, gridOpacity=.14).configure_view(stroke=None))
    return panels


def brightness_chart(rows: list[dict]):
    """Os dois gráficos oficiais de brilho; passar o mouse num ponto mostra o corte."""
    data = pd.DataFrame({
        "case_id": [row["case_id"] for row in rows],
        "corte": [case_name(row["case_id"]) for row in rows],
        "ordem": list(range(1, len(rows) + 1)),
        "delta_mu": [number(row, "delta_mu") for row in rows],
        "sat31": [number(row, "sat31_pct") for row in rows],
        "decisão": [official_brightness_status(row) for row in rows],
    })
    base = alt.Chart(data).mark_circle(size=75, color="#0d6b61").encode(
        x=alt.X("ordem:Q", title="cortes do dataset"),
        tooltip=[alt.Tooltip("corte:N"), alt.Tooltip("delta_mu:Q", title="Δμ", format="+.3f"), alt.Tooltip("sat31:Q", title="Sat31 (%)", format=".2f"), alt.Tooltip("decisão:N")],
        opacity=alt.value(.8),
    )
    review_rules = alt.Chart(pd.DataFrame({"limite": [-OFFICIAL_REVIEW, OFFICIAL_REVIEW]})).mark_rule(color="#8a5a22", strokeDash=[4, 4]).encode(y=alt.Y("limite:Q"))
    reject_rules = alt.Chart(pd.DataFrame({"limite": [-OFFICIAL_REJECT, OFFICIAL_REJECT]})).mark_rule(color="#b42318", strokeDash=[4, 4]).encode(y=alt.Y("limite:Q"))
    sat_rule = alt.Chart(pd.DataFrame({"limite": [OFFICIAL_SATURATION]})).mark_rule(color="#b42318", strokeDash=[4, 4]).encode(y="limite:Q")
    delta = (review_rules + reject_rules + base.encode(y=alt.Y("delta_mu:Q", title="Δμ · sintética − real"))).properties(title="Δμ por corte", height=280)
    saturation = (sat_rule + base.encode(y=alt.Y("sat31:Q", title="Sat31 (%)"))).properties(title="Saturação no tom 31", height=280)
    return [chart.configure_axis(grid=True, gridOpacity=.14).configure_view(stroke=None) for chart in (delta, saturation)]


def draw_charts(charts: list, per_row: int = 2) -> None:
    for start in range(0, len(charts), per_row):
        for column, chart in zip(st.columns(per_row), charts[start:start + per_row]):
            column.altair_chart(chart, use_container_width=True)


def metrics_table(rows: list[dict], mode: str) -> pd.DataFrame:
    if mode == "brilho":
        data = [{"case_id": row["case_id"], "corte": case_name(row["case_id"]), "decisão oficial": official_brightness_status(row), "por quê": brightness_explanation(row), "|Δμ|": number(row, "abs_delta_mu"), "Sat31 (%)": number(row, "sat31_pct"), "Wavelet LL |Δ|": abs(number(row, "wavelet_ll_synthetic") - number(row, "wavelet_ll_real"))} for row in rows]
    else:
        data = [{"case_id": row["case_id"], "corte": case_name(row["case_id"]), "leitura exploratória": row.get("wavelet_verdict", "n/a"), "o que mudou": texture_explanation(row), "GLCM contraste (%)": number(row, "glcm_contrast_pct"), "GLCM energia (%)": 100 * number(row, "glcm_energy_synthetic") / number(row, "glcm_energy_real"), "Wavelet 2 px (%)": number(row, "wavelet_fine_pct"), "Wavelet 4 px (%)": number(row, "wavelet_coarse_pct"), "Wavelet entropia Δ": number(row, "wavelet_entropy_synthetic") - number(row, "wavelet_entropy_real")} for row in rows]
    frame = pd.DataFrame(data)
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].round(3)
    return frame


def show_method_cards(mode: str):
    if mode == "brilho":
        st.markdown("#### O que decide o filtro de brilho")
        st.caption("A decisão oficial sempre considera as duas medidas: diferença de tom médio (|Δμ|) e porcentagem de pixels no tom 31 (Sat31).")
        first, second, third = st.columns(3)
        first.metric("Aprovada", f"|Δμ| ≤ {br(OFFICIAL_REVIEW)}", "Sat31 ≤ 15% · ambas dentro da faixa")
        second.metric("Revisão manual", f"|Δμ| em {br(OFFICIAL_REVIEW)}–{br(OFFICIAL_REJECT)}", "Sat31 ≤ 15% · Δμ pede inspeção")
        third.metric("Rejeição direta", f"|Δμ| > {br(OFFICIAL_REJECT)}", "ou Sat31 > 15%")
        st.caption(f"Os limites são 1 e 2 IQR das médias pulmonares das CTs reais de referência ({br(OFFICIAL_REVIEW)} nesta calibração).")
        st.info("Wavelet LL ajuda a interpretar a variação de brilho em baixa frequência no corte. É complementar: não muda a categoria oficial, que depende somente de |Δμ| e Sat31.")
    else:
        st.markdown("#### Técnicas de textura usadas")
        st.dataframe(pd.DataFrame([
            ["GLCM · contraste", "quanto o tom muda entre pixels vizinhos; queda sugere suavização/perda de alta frequência"],
            ["GLCM · energia", "uniformidade dos pares de tons; pode subir quando a imagem fica mais lisa"],
            ["Wavelet · energia (2 + 4 px)", "quantidade total de detalhe, somando a escala fina (~2 px) e a grossa (~4 px); na tabela, as duas escalas aparecem separadas"],
            ["Wavelet · entropia", "variação/complexidade da distribuição de detalhes"],
        ], columns=["técnica", "o que ela mede"]), hide_index=True, width="stretch")
        st.warning("Vereditos como “borrou” são exploratórios: ajudam a descrever e priorizar inspeção, mas não são o gatekeeper oficial de brilho nem uma decisão clínica.")


def show_case(board: dict, history_dir: Path, stamp: float, case_id: str, version: str, mode: str):
    row = next(item for item in board["rows"][version] if item["case_id"] == case_id)
    st.markdown("---")
    st.subheader(f"Par em foco · {case_name(case_id)}")
    if mode == "brilho":
        st.markdown(f"**Gatekeeper oficial: {official_brightness_status(row)}.** {brightness_explanation(row)}")
        choices = {"CT real × sintética (Δμ)": "delta_mu", "Pixels extremos (contexto exploratório)": "densidade", "Brilho local (Wavelet LL)": "ll"}
    else:
        st.markdown(f"**Leitura de textura: {row.get('wavelet_verdict', 'n/a')}.** {texture_explanation(row)}")
        choices = {"GLCM · contraste": "glcm_contraste", "Wavelet · detalhe fino": "wavelet_fina", "Wavelet · detalhe grosso": "wavelet_grossa"}
    st.image(raw_pair_png(str(history_dir), stamp, case_id, version), use_container_width=True)
    st.caption("CTs brutas lado a lado — sem mapa, máscara ou métrica sobreposta. A mesma janela de visualização é usada nas duas imagens para permitir comparação direta.")
    tones = glcm_tone_counts(str(history_dir), stamp, case_id, version)
    st.markdown("#### Extremos da discretização GLCM")
    metrics = st.columns(4)
    for column, label, key in zip(metrics, ("CT real · tom 0", f"CT real · tom {tones['last_tone']}", "CT sintética · tom 0", f"CT sintética · tom {tones['last_tone']}"), ("real_zero", "real_last", "synthetic_zero", "synthetic_last")):
        _, pct = tones[key]
        column.metric(label, f"{pct:.2f}%", "dos pixels da ROI")
    st.caption(f"A régua oficial discretiza a ROI pulmonar em {tones['last_tone'] + 1} tons. Os cartões mostram a porcentagem no menor tom (0) e no maior tom ({tones['last_tone']}); não contam pares de vizinhos da GLCM.")
    with st.expander("Entender a métrica neste par", expanded=False):
        st.caption("Esta visualização mostra o que a técnica calcula, não a CT bruta acima.")
        technique = st.segmented_control("Técnica", list(choices), default=list(choices)[0], key=f"technique_{mode}")
        st.image(technique_png(str(history_dir), stamp, case_id, choices[technique]), use_container_width=True)
        st.caption(TECHNIQUE_FIGURES[choices[technique]])
    with st.expander("Mapa regional — onde a diferença aparece", expanded=False):
        st.caption("Use este mapa apenas para localizar um padrão já visto no par. Ele divide o pulmão em blocos de 16 px; não é uma nova regra de aprovação.")
        columns = st.columns(2)
        default_mode = "Comparado com a original" if mode == "textura" else "Valor de cada imagem"
        map_mode = columns[0].segmented_control("Mostrar", ["Valor de cada imagem", "Comparado com a original"], default=default_mode, key=f"map_mode_{mode}")
        compared = map_mode == "Comparado com a original"
        layers = {label: key for key, label in MAP_LAYERS.items() if compared or key != "veredito"}
        layer = columns[1].segmented_control("Camada", list(layers), default=list(layers)[0], key=f"map_layer_{mode}_{compared}")
        st.image(map_png(str(history_dir), stamp, case_id, layers[layer], compared), use_container_width=True)
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
        selection = st.selectbox("Mostrar cortes", ["todos", "aprovada", "revisão manual", "rejeição direta"], key="case_filter_brilho")
        choices = [row["case_id"] for row in rows if selection == "todos" or official_brightness_status(row) == selection]
    else:
        selection = st.selectbox("Mostrar cortes", ["todos", *sorted({row.get("wavelet_verdict", "n/a") for row in rows})], key="case_filter_textura")
        choices = [row["case_id"] for row in rows if selection == "todos" or row.get("wavelet_verdict") == selection]
    case_id = st.selectbox("Corte para inspecionar", choices, format_func=case_name, key="case_selector")

show_method_cards(mode)
if mode == "brilho":
    statuses = pd.Series([official_brightness_status(row) for row in rows]).value_counts()
    first, second, third = st.columns(3)
    first.metric("Aprovadas pelo gatekeeper", int(statuses.get("aprovada", 0)), f"de {len(rows)} cortes")
    second.metric("Para revisão manual", int(statuses.get("revisão manual", 0)), "regra oficial")
    third.metric("Rejeições diretas", int(statuses.get("rejeição direta", 0)), "regra oficial")
else:
    below_identity = sum(number(row, "glcm_contrast_pct") < 100 for row in rows)
    median_contrast = np.median([number(row, "glcm_contrast_pct") for row in rows])
    first, second, third = st.columns(3)
    first.metric("Contraste GLCM abaixo da identidade", f"{below_identity}/{len(rows)}", "sintética < real")
    second.metric("Contraste GLCM mediano", f"{median_contrast:.0f}%", "sintética / real")
    third.metric("Cortes marcados “borrou”", sum(row.get("wavelet_verdict") == "borrou" for row in rows), "leitura exploratória")

if mode == "textura":
    st.markdown("#### Relação entre CT real e sintética")
    scale_glcm, scale_wavelet = st.columns(2)
    glcm_choice = scale_glcm.segmented_control("Escala · energia GLCM", ["Automática", "Até 0,10", "Até 0,20"], default="Automática", help="Use a mesma régua ao alternar Dsr e Dgen.")
    wavelet_choice = scale_wavelet.segmented_control("Escala · energia Wavelet", ["Automática", "Até 0,01", "Até 0,02", "Até 0,05"], default="Automática", help="Wavelet usa valores muito menores do que energia GLCM; esta escolha não altera o gráfico GLCM.")
    glcm_ceiling = {"Automática": None, "Até 0,10": .10, "Até 0,20": .20}[glcm_choice]
    wavelet_ceiling = {"Automática": None, "Até 0,01": .01, "Até 0,02": .02, "Até 0,05": .05}[wavelet_choice]
    st.caption("Passe o mouse num ponto para ver o corte. A linha tracejada é a identidade: real = sintética. Para abrir um corte, clique na linha dele na tabela abaixo.")
    draw_charts(texture_chart(rows, glcm_ceiling, wavelet_ceiling))
else:
    st.markdown("#### Resposta do gatekeeper de brilho")
    st.caption("Passe o mouse num ponto para ver o corte. Linhas tracejadas indicam os limites oficiais de revisão e rejeição. Para abrir um corte, clique na linha dele na tabela abaixo.")
    draw_charts(brightness_chart(rows))
with st.expander("Ver todos os cortes e o motivo", expanded=True):
    table = metrics_table(rows, mode)
    # reset_index: a linha clicada volta como posição, e ela precisa bater com a ordem mostrada
    table = table.sort_values(["decisão oficial", "|Δμ|"] if mode == "brilho" else "GLCM contraste (%)", ascending=[True, False] if mode == "brilho" else True).reset_index(drop=True)
    table.insert(0, "em foco", np.where(table["case_id"] == case_id, "●", ""))
    st.caption("Clique numa linha para abrir o corte abaixo.")
    event = st.dataframe(table, hide_index=True, width="stretch", height=320, column_order=[column for column in table.columns if column != "case_id"],
                         on_select="rerun", selection_mode="single-row", key=f"cases_{mode}")
    selected = event.selection.rows
    clicked = table.iloc[selected[0]]["case_id"] if selected else None
    # a seleção continua ativa nas próximas execuções: só um clique novo muda o corte,
    # senão a escolha feita na barra lateral seria desfeita
    if clicked and clicked != st.session_state.get(f"cases_{mode}_last") and clicked != case_id:
        st.session_state[f"cases_{mode}_last"] = clicked
        st.session_state["pending_case"] = clicked
        st.session_state["pending_mode"] = mode
        st.rerun()
    st.session_state[f"cases_{mode}_last"] = clicked
    if mode == "brilho":
        st.caption("“Decisão oficial” usa somente |Δμ| e Sat31. Wavelet LL aparece para contexto, mas não altera a categoria.")
    else:
        st.caption("“Leitura exploratória” descreve a alteração de textura observada; não é aprovação/rejeição clínica. Os valores percentuais comparam sintética ÷ real no mesmo corte.")
    st.caption("Os pontos e linhas resumem cortes do dataset; fases do mesmo paciente não devem ser interpretadas como pacientes independentes.")

show_case(board, history_dir, stamp, case_id, version, mode)
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
