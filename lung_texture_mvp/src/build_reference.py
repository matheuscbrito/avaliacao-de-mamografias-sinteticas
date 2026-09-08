"""Construção da referência global a partir de originais aceitas."""
from __future__ import annotations

import numpy as np


FEATURES = ("contrast", "energy")


def build_reference(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Gera estatísticas descritivas e limites de revisão Tukey por feature."""
    reference: dict[str, dict[str, float]] = {}
    for feature in FEATURES:
        values = np.array([float(row[feature]) for row in rows], dtype=float)
        q1, q3 = np.percentile(values, (25, 75))
        iqr = q3 - q1
        reference[feature] = {
            "n": int(values.size), "mean": float(values.mean()), "median": float(np.median(values)),
            "std": float(values.std(ddof=0)), "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
            "min": float(values.min()), "max": float(values.max()), "p5": float(np.percentile(values, 5)), "p95": float(np.percentile(values, 95)),
            "tukey_lower": float(q1 - 1.5 * iqr), "tukey_upper": float(q3 + 1.5 * iqr),
        }
    return reference


def reference_rows(reference: dict[str, dict[str, float]]) -> list[dict]:
    return [{"feature": feature, **stats} for feature, stats in reference.items()]
