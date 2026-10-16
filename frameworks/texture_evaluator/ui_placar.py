"""Interface do placar por versão: adicionar versões e ver brilho e textura ao longo do tempo.

Na raiz do repositório:
    .venv/bin/streamlit run frameworks/texture_evaluator/ui_placar.py --theme.base light --theme.font monospace

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
    CHANGE_NOTE, GROUPS, INDICATORS, VERSION_HEADER, PlacarError, add_version, build_board, case_map, change_table,
    count_table, figure, load_history, remove_version, score_table, version_table, write_outputs,
)
from frameworks.texture_evaluator.regional import MASK_ONLY  # noqa: E402

# colunas da tabela por corte, por seção
CASE_COLUMNS = {
    "final": ["final_status", "final_reasons", "pair_status", "pair_correlation"],
    "brilho": ["abs_delta_mu", "brightness_status", "density_status", "map_area_mais_claro_pct", "map_area_mais_escuro_pct"],
    "textura": ["wavelet_verdict", "wavelet_fine_pct", "wavelet_coarse_pct", "wavelet_ratio", "glcm_contrast_pct", "map_area_textura_alterada_pct", "map_corr_textura"],
}
GROUP_NOTES = {
    "final": "Rejeitar: par suspeito ou filtro de brilho do contrato. Revisar: qualquer sinal de brilho ou de textura. Aprovada: nada disso.",
    "brilho": "Filtro do contrato (Δμ, Sat31) e densidade (pixels fora da faixa P1–P99 real) lado a lado; o mapa mede brilho bloco a bloco (16 px).",
    "textura": "Detalhe fino e grosso em % da original (Wavelet), contraste GLCM e, no mapa, a área do pulmão com textura alterada e se a textura está nos mesmos lugares.",
}


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


def table(rows, header):
    st.table(pd.DataFrame(rows, columns=header).set_index(header[0]))


st.set_page_config(page_title="Placar de textura", layout="wide")
st.title("Placar por versão do gerador")

history_dir = resolve(st.text_input("Pasta do histórico", "outputs/placar"))
history = load_history(history_dir)

# ---------------------------------------------------------------- adicionar versão
st.header("Adicionar versão")
if history is None:
    st.markdown("Histórico vazio. A primeira versão adicionada vira a referência: as réguas são calibradas nas originais dela.")
else:
    st.markdown(f"Referência: {history['reference']} · máscaras: {short(history['mask_dir']) if history['mask_dir'] else 'junto das imagens'} · versões: {len(history['versions'])}")
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
pairs = [f"{a} → {b}" for a, b in board["comparisons"]]

# ---------------------------------------------------------------- versões
st.header("Versões")
st.markdown(
    "Pareamento pelo nome: X.tiff ↔ X_generate.tiff ↔ X_mask.tiff. "
    "Modo mesma anatomia: a sintética parte da própria original; pares com correlação < 0,5 com a original são rejeitados como suspeitos (arquivo trocado?). "
    "Modo só máscara: a sintética é outro pulmão no mesmo contorno; o par não pode ser conferido e o mapa por região não se aplica."
)
table([[row[0], short(row[1]), *row[2:]] for row in version_table(board)], VERSION_HEADER)
with st.expander("Remover versão"):
    target = st.selectbox("Versão a remover", names, key="remove")
    if st.button("Remover"):
        remove_version(history_dir, target)
        board = build_board(history_dir)
        if board:
            write_outputs(history_dir, board)
        st.rerun()
st.markdown(f"{board['n_cases']} cortes comuns a todas as versões · mediana (ou % dos cortes) [intervalo de 95%, bootstrap] · ideal = sintética igual à original")

# ---------------------------------------------------------------- seções
for group, title in GROUPS.items():
    st.header(title)
    st.markdown(GROUP_NOTES[group])
    table(score_table(board, group), ["Indicador", "Ideal", *names])
    if pairs:
        table(change_table(board, group), ["Mudança", *pairs])
    table(count_table(board, group), ["Contagem de cortes", *names])
    if len(names) > 1:
        fig = figure(board, group)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
if pairs:
    st.caption(CHANGE_NOTE)

# ---------------------------------------------------------------- por corte
st.header("Por corte")
columns = st.columns([1, 1, 1, 2])
version = columns[0].selectbox("Versão", names, index=len(names) - 1)
section = columns[1].selectbox("Colunas", list(GROUPS), format_func=GROUPS.get)
status = columns[2].selectbox("Veredito final", ["todos", "aprovada", "revisar", "rejeitar"])
sortable = {indicator.label: indicator.key for indicator in INDICATORS if indicator.key in CASE_COLUMNS[section]}
order = columns[3].selectbox("Ordenar por", ["corte", *sortable])
cases = pd.DataFrame(board["rows"][version])[["case_id", *dict.fromkeys(CASE_COLUMNS["final"][:1] + CASE_COLUMNS[section])]]
for column in cases.columns:
    numeric = pd.to_numeric(cases[column], errors="coerce")
    if numeric.notna().any():
        cases[column] = numeric.round(3)
if status != "todos":
    cases = cases[cases["final_status"] == status]
cases = cases.sort_values("case_id" if order == "corte" else sortable[order])
st.dataframe(cases, hide_index=True, width="stretch")

st.subheader("Mapa por região")
entry = next(v for v in board["history"]["versions"] if v["name"] == version)
if entry.get("mode") == MASK_ONLY:
    st.markdown(f"{version} está no modo só máscara: a sintética não tem a mesma anatomia da original, então o mapa por região não se aplica.")
elif not len(cases):
    st.markdown("Nenhum corte com esse filtro.")
else:
    case_id = st.selectbox("Corte", list(cases["case_id"]))
    row = next(r for r in board["rows"][version] if r["case_id"] == case_id)
    st.markdown(f"Veredito: {row['final_status']} · {row['final_reasons']}")
    try:
        fig = case_map(history_dir, version, case_id)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    except PlacarError as error:
        st.error(str(error))
    st.markdown("Blocos de 16 px com ≥ 90% de pulmão. Brilho: deslocamento da média do bloco em relação ao mesmo bloco da original (vermelho = mais claro, azul = mais escuro). Textura: veredito Wavelet do bloco.")

st.markdown(f"Arquivos em {short(history_dir)}: placar.md, placar.csv, comparacao.csv, por_caso.csv")
