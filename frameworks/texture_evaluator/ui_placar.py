"""Interface do placar por versão: cada técnica com seu score e a imagem do que ela enxerga.

Na raiz do repositório:
    .venv/bin/streamlit run frameworks/texture_evaluator/ui_placar.py --theme.base light --theme.font monospace

Tudo o que a interface mostra vem do histórico em disco (`historico.json` e
`versoes/*.csv`), o mesmo que a linha de comando `placar.py` usa.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from frameworks.texture_evaluator.placar import (  # noqa: E402
    CHANGE_NOTE, GROUPS, INDICATORS, VERSION_HEADER, PlacarError, add_version, build_board, count_table, figure,
    load_history, remove_version, technique_table, version_table, write_outputs,
)
from frameworks.texture_evaluator.regional import MASK_ONLY, PAIR_MIN_CORRELATION  # noqa: E402
from frameworks.texture_evaluator.visual import MAP_LAYERS, TECHNIQUE_FIGURES, map_figure, technique_figure  # noqa: E402

CASE_COLUMNS = ["case_id", "final_status", "final_reasons", "pair_status", "abs_delta_mu", "density_status", "wavelet_verdict",
                "wavelet_fine_pct", "wavelet_coarse_pct", "wavelet_ratio", "glcm_contrast_pct", "map_area_textura_alterada_pct", "map_corr_textura"]


def short(path) -> str:
    """Caminho relativo ao repositório quando possível, para não alargar a página."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve(text: str) -> Path | None:
    if not text.strip():
        return None
    return Path(text) if Path(text).is_absolute() else ROOT / text


def case_name(case_id: str) -> str:
    """Nome curto do corte: fase do ciclo e número do corte (ex.: 30.0A · 053)."""
    phase = re.search(r"Gated ([\d.]+A)", case_id)
    return f"{phase.group(1)} · {case_id.rsplit('-', 1)[-1]}" if phase else case_id


def png(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


# as figuras dependem só do histórico (versão gravada em disco) e das escolhas: guardar evita redesenhar a cada clique
@st.cache_data(show_spinner="Desenhando...")
def technique_png(history_dir: str, stamp: float, case_id: str, technique: str) -> bytes:
    board = build_board(Path(history_dir))
    rows = {name: next(r for r in board["rows"][name] if r["case_id"] == case_id) for name in board["names"]}
    return png(technique_figure(Path(history_dir), case_id, technique, rows))


@st.cache_data(show_spinner="Desenhando...")
def map_png(history_dir: str, stamp: float, case_id: str, layer: str, compared: bool) -> bytes:
    return png(map_figure(Path(history_dir), case_id, layer, compared))


def show_table(header, rows):
    st.dataframe(pd.DataFrame(rows, columns=header), hide_index=True, width="stretch")


st.set_page_config(page_title="Placar de textura", layout="wide")

# ---------------------------------------------------------------- barra lateral: histórico e corte
history_dir = resolve(st.sidebar.text_input("Pasta do histórico", "outputs/placar"))
history = load_history(history_dir)
stamp = (history_dir / "historico.json").stat().st_mtime if history else 0.0

st.title("Placar por versão do gerador")

with st.expander("Versões no histórico · adicionar ou remover", expanded=not history):
    if history:
        st.markdown(f"Referência das réguas: {history['reference']} · máscaras: {short(history['mask_dir']) if history['mask_dir'] else 'junto das imagens'}")
    else:
        st.markdown("Histórico vazio. A primeira versão adicionada vira a referência: as réguas são calibradas nas originais dela.")
    with st.form("add"):
        columns = st.columns([1, 3, 3])
        name = columns[0].text_input("Nome", placeholder="v3")
        folder = columns[1].text_input("Pasta com originais e sintéticas (_generate)", placeholder="CAMINHO/DA/VERSAO")
        masks = columns[2].text_input("Pasta das máscaras (vazio = a do histórico)", "" if history else "dados/Dgen")
        submitted = st.form_submit_button("Avaliar e adicionar")
    if submitted:
        try:
            with st.spinner("Avaliando..."):
                message = add_version(history_dir, name, resolve(folder), resolve(masks))
                write_outputs(history_dir, build_board(history_dir))
            st.session_state["message"] = message
            st.rerun()
        except PlacarError as error:
            st.error(str(error))
    if history and history["versions"]:
        columns = st.columns([2, 1])
        target = columns[0].selectbox("Versão a remover", [v["name"] for v in history["versions"]], label_visibility="collapsed")
        if columns[1].button("Remover versão"):
            remove_version(history_dir, target)
            board = build_board(history_dir)
            if board:
                write_outputs(history_dir, board)
            st.rerun()
if "message" in st.session_state:
    st.markdown(st.session_state.pop("message"))

try:
    board = build_board(history_dir)
except PlacarError as error:
    st.error(str(error))
    st.stop()
if board is None:
    st.stop()
names = board["names"]
latest = {row["case_id"]: row for row in board["rows"][names[-1]]}

st.sidebar.markdown("---")
status = st.sidebar.selectbox(f"Cortes com veredito ({names[-1]})", ["todos", "aprovada", "revisar", "rejeitar"])
choices = [case for case in latest if status == "todos" or latest[case]["final_status"] == status]
if not choices:
    st.sidebar.markdown("Nenhum corte com esse veredito.")
    st.stop()
case_id = st.sidebar.selectbox("Corte para inspecionar", choices, format_func=lambda case: f"{case_name(case)} · {latest[case]['final_status']}")
st.sidebar.markdown(f"{case_id}")
for name in names:
    row = next(r for r in board["rows"][name] if r["case_id"] == case_id)
    st.sidebar.markdown(f"**{name}**: {row['final_status']} · {row['final_reasons']}")

# ---------------------------------------------------------------- versões
st.markdown(f"{len(names)} versões · {board['n_cases']} cortes comuns a todas · score = mediana entre os cortes (ou % dos cortes) [intervalo de 95%]")
show_table(VERSION_HEADER, [[row[0], short(row[1]), *row[2:]] for row in version_table(board)])
with st.expander("Como o par original × sintética é conferido, e o que é o modo só máscara"):
    st.markdown(
        f"""
- **Pareamento pelo nome:** `X.tiff` (original) ↔ `X_generate.tiff` (sintética) ↔ `X_mask.tiff` (máscara). As originais de todas as versões precisam ser idênticas às da referência.
- **Mesma anatomia** (ex.: Dsr, gerado a partir da própria original em baixa resolução): a sintética é o mesmo pulmão. O framework confere o par pela correlação dos pixels (pares verdadeiros ficam ≥ 0,80); abaixo de {PAIR_MIN_CORRELATION} o par é rejeitado como suspeito de arquivo trocado.
- **Só máscara** (ex.: Dgen, gerado só a partir do contorno do pulmão): a sintética é **outro pulmão** dentro do mesmo contorno. Os números globais (brilho médio, quantidade de textura fina e grossa, GLCM) valem, porque comparam o pulmão inteiro. Já o que compara **ponto a ponto** não vale: o bloco (x, y) da sintética não é o mesmo tecido do bloco (x, y) da original, então a área alterada e a correlação espacial ficam como n/a, e o mapa só mostra o **valor de cada imagem**, não a comparação.
"""
    )

# ---------------------------------------------------------------- técnicas
for group, title in GROUPS.items():
    st.header(title)
    show_table(*technique_table(board, group))
    with_image = [indicator for indicator in INDICATORS if indicator.group == group and indicator.image]
    if with_image:
        options = {indicator.label: indicator.image for indicator in with_image}
        chosen = st.segmented_control(f"Ver o que a técnica enxerga no corte {case_name(case_id)}", list(options), key=f"image_{group}")
        if chosen:
            st.image(technique_png(str(history_dir), stamp, case_id, options[chosen]))
            st.caption(TECHNIQUE_FIGURES[options[chosen]])
    with st.expander("Contagem de cortes por categoria" + (" e evolução ao longo das versões" if len(names) > 1 else "")):
        show_table(["Categoria", *names], count_table(board, group))
        if len(names) > 1:
            st.image(png(figure(board, group)))
if board["comparisons"]:
    st.caption(CHANGE_NOTE)

# ---------------------------------------------------------------- mapa
st.header("Mapa por região")
st.markdown(f"Corte {case_name(case_id)}, em blocos de 16 px com ≥ 90% de pulmão.")
columns = st.columns(2)
mode = columns[0].segmented_control("Mostrar", ["Valor de cada imagem", "Comparado com a original"], default="Comparado com a original", key="map_mode")
compared = mode == "Comparado com a original"
layers = {label: key for key, label in MAP_LAYERS.items() if compared or key != "veredito"}
layer = columns[1].segmented_control("Camada", list(layers), default=list(layers)[0], key=f"map_layer_{compared}")
if mode and layer:
    st.image(map_png(str(history_dir), stamp, case_id, layers[layer], compared))
    if compared:
        st.caption("Cada bloco da sintética dividido pelo mesmo bloco da original. Só vale para versões de mesma anatomia." + (
            " Versões só máscara aparecem sem mapa: veja 'Valor de cada imagem' para elas." if any(v.get("mode") == MASK_ONLY for v in board["history"]["versions"]) else ""))
    else:
        st.caption("O mapa de cada imagem sozinha, na mesma escala. Vale para todas as versões, inclusive só máscara: compare o desenho e a intensidade, não bloco com bloco.")

# ---------------------------------------------------------------- todos os cortes
st.header("Todos os cortes")
version = st.selectbox("Versão", names, index=len(names) - 1)
cases = pd.DataFrame(board["rows"][version])[CASE_COLUMNS]
cases.insert(1, "corte", cases["case_id"].map(case_name))
for column in cases.columns:
    numeric = pd.to_numeric(cases[column], errors="coerce")
    if numeric.notna().any():
        cases[column] = numeric.round(3)
st.dataframe(cases, hide_index=True, width="stretch")
st.markdown(f"Arquivos em {short(history_dir)}: placar.md, placar.csv, comparacao.csv, por_caso.csv")
