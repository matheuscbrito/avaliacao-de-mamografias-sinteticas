"""Textos e utilitários compartilhados pelas páginas do dashboard."""
from __future__ import annotations

import csv
import html
from pathlib import Path

import numpy as np
import streamlit as st

FEATURES = {"energy": "Energia", "entropy": "Entropia"}
DECISIONS = {
    "approved": ("Aprovada", "#16803c", "Pode entrar no conjunto de data augmentation."),
    "approved_with_observation": ("Aprovada com observação", "#277da1", "Caso raro, mas compatível com a original."),
    "manual_review": ("Revisão manual", "#b76e00", "Ainda não entra no augmentation: vale olhar este par com calma."),
    "reject": ("Não aprovada", "#c73e1d", "Não deve entrar no data augmentation."),
}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as file:
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


def decision_card(status: str, reason: str, heading: str = "") -> None:
    title, colour, meaning = DECISIONS[status]
    heading_html = f'<div style="font-size:.85rem;color:#555;margin-bottom:2px">{html.escape(heading)}</div>' if heading else ""
    st.markdown(
        f"""<div style="border-left:7px solid {colour};background:#f8f9fb;color:#1f2328;border-radius:8px;padding:16px 18px;margin:4px 0 18px">
        {heading_html}<div style="font-size:1.15rem;font-weight:700;color:{colour}">{title}</div>
        <div style="margin-top:5px">{html.escape(meaning)}</div>
        <div style="margin-top:7px;color:#555">Motivo: {html.escape(reason)}</div></div>""",
        unsafe_allow_html=True,
    )
