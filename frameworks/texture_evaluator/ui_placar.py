"""Interface do placar por versão: adicionar versões e ver os scores ao longo do tempo.

Na raiz do repositório:
    .venv/bin/streamlit run frameworks/texture_evaluator/ui_placar.py

Tudo o que a interface mostra vem do histórico em disco (`historico.json` e
`versoes/*.csv`), o mesmo que a linha de comando `placar.py` usa.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from frameworks.texture_evaluator.placar import (  # noqa: E402
    MIN_CHANGE, PlacarError, add_version, all_indicators, build_board, change_table, figure,
    load_history, remove_version, score_table, verdict_table, write_outputs,
)

CASE_COLUMNS = ["case_id", "final_status", "final_reasons", "wavelet_fine_pct", "wavelet_coarse_pct", "wavelet_ratio", "glcm_contrast_pct", "abs_delta_mu", "brightness_status", "density_status"]

st.set_page_config(page_title="Placar de textura", layout="wide")
st.title("Placar de textura por versão")

history_dir = Path(st.text_input("Pasta do histórico", "outputs/placar"))
if not history_dir.is_absolute():
    history_dir = ROOT / history_dir
history = load_history(history_dir)


def short(path) -> str:
    """Caminho relativo ao repositório quando possível, para não alargar a página."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------- adicionar versão
st.subheader("Adicionar versão")
if history is None:
    st.markdown("Histórico vazio. A primeira versão adicionada vira a referência: as réguas são calibradas nas originais dela.")
else:
    st.markdown(f"Referência: {history['reference']} · máscaras: {short(history['mask_dir']) if history['mask_dir'] else 'junto das imagens'} · versões: {len(history['versions'])}")
with st.form("add", clear_on_submit=False):
    columns = st.columns([1, 3, 3])
    name = columns[0].text_input("Nome", placeholder="v3")
    folder = columns[1].text_input("Pasta com originais e sintéticas (_generate)", placeholder="CAMINHO/DA/VERSAO")
    masks = columns[2].text_input("Pasta das máscaras (vazio = a do histórico)", "" if history else "dados/Dgen")
    submitted = st.form_submit_button("Avaliar e adicionar")
if submitted:
    try:
        resolve = lambda text: (ROOT / text if not Path(text).is_absolute() else Path(text)) if text.strip() else None  # noqa: E731
        with st.spinner("Avaliando..."):
            message = add_version(history_dir, name, resolve(folder), resolve(masks))
            board = build_board(history_dir)
            write_outputs(history_dir, board)
        # recarrega a página para o cabeçalho e as tabelas mostrarem o histórico novo
        st.session_state["message"] = message
        st.rerun()
    except PlacarError as error:
        st.error(str(error))
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

# ---------------------------------------------------------------- scores
st.subheader("Score por versão")
st.markdown(f"{board['n_cases']} cortes comuns a todas as versões · mediana [intervalo de 95%, bootstrap] · ideal = sintética igual à original")
st.table(pd.DataFrame(score_table(board), columns=["Indicador", "Ideal", *names]).set_index("Indicador"))

if board["comparisons"]:
    st.subheader("Mudança em relação à versão anterior")
    st.markdown(
        "Ganho = quanto, corte a corte, a versão chegou mais perto do ideal (mediana; taxas em pontos percentuais).  \n"
        "melhorou / piorou: intervalo de 95% fora do zero e mudança acima do mínimo "
        f"({MIN_CHANGE['wavelet_fine_pct']:.0f} pp; {MIN_CHANGE['wavelet_ratio']} na razão e no |Δμ|). "
        "mudança pequena: fora do zero, abaixo do mínimo."
    )
    st.table(pd.DataFrame(change_table(board), columns=["Indicador", *(f"{a} → {b}" for a, b in board["comparisons"])]).set_index("Indicador"))

st.subheader("Ao longo das versões")
fig = figure(board)
st.pyplot(fig, width="stretch")
plt.close(fig)

st.subheader("Vereditos (número de cortes)")
st.table(pd.DataFrame(verdict_table(board), columns=["Veredito", *names]).set_index("Veredito"))

# ---------------------------------------------------------------- por corte
st.subheader("Por corte")
columns = st.columns([1, 1, 2])
version = columns[0].selectbox("Versão", names, index=len(names) - 1)
status = columns[1].selectbox("Veredito final", ["todos", "aprovada", "revisar", "rejeitar"])
indicator_names = {label: key for key, label, *_ in all_indicators() if key in CASE_COLUMNS}
order = columns[2].selectbox("Ordenar por", list(indicator_names), index=0)
cases = pd.DataFrame(board["rows"][version])[CASE_COLUMNS]
for column in CASE_COLUMNS[3:8]:
    cases[column] = cases[column].astype(float).round(3)
if status != "todos":
    cases = cases[cases["final_status"] == status]
st.dataframe(cases.sort_values(indicator_names[order]), hide_index=True, width="stretch")

# ---------------------------------------------------------------- versões
st.subheader("Versões no histórico")
st.table(pd.DataFrame([[v["name"], short(v["folder"]), v["added"].replace("T", " "), v["n_cases"]] for v in board["history"]["versions"]], columns=["Versão", "Pasta", "Adicionada em", "Cortes válidos"]).set_index("Versão"))
with st.expander("Remover versão"):
    target = st.selectbox("Versão a remover", names, key="remove")
    if st.button("Remover"):
        remove_version(history_dir, target)
        board = build_board(history_dir)
        if board:
            write_outputs(history_dir, board)
        st.rerun()
st.markdown(f"Arquivos em {short(history_dir)}: placar.md, placar.csv, comparacao.csv, por_caso.csv")
