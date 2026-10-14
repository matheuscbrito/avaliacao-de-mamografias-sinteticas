"""Comparação da sintética com o par e com a distribuição de originais."""
from __future__ import annotations

from .build_reference import FEATURES


def _outside(value: float, stats: dict[str, float]) -> bool:
    return value < stats["tukey_lower"] or value > stats["tukey_upper"]


def compare_case(case_id: str, original: dict, synthetic: dict, reference: dict[str, dict[str, float]], pair_iqr_threshold: float = 1.0) -> dict:
    """Combina as duas perguntas do framework em uma classificação inicial.

    A distância do par é normalizada pelo IQR das originais. Assim, contraste e
    energia são comparados na escala da variabilidade real de cada feature.
    O limiar de 1 IQR é uma configuração de MVP, não um limiar clínico.
    """
    row: dict[str, float | str | bool] = {"case_id": case_id, "pair_iqr_threshold": pair_iqr_threshold}
    pair_distant, synthetic_outside = False, False
    original_outside_same_direction = True
    for feature in FEATURES:
        original_value, synthetic_value = float(original[feature]), float(synthetic[feature])
        delta = synthetic_value - original_value
        stats = reference[feature]
        relative_delta = abs(delta) / max(abs(original_value), 1e-12)
        pair_distance_iqr = abs(delta) / max(stats["iqr"], 1e-12)
        is_outside = _outside(synthetic_value, stats)
        original_outside = _outside(original_value, stats)
        same_direction = (synthetic_value - stats["median"]) * (original_value - stats["median"]) >= 0
        row.update({
            f"original_{feature}": original_value, f"synthetic_{feature}": synthetic_value,
            f"delta_{feature}": delta, f"relative_delta_{feature}": relative_delta, f"pair_distance_{feature}_iqr": pair_distance_iqr,
            f"synthetic_{feature}_outside_reference": is_outside,
            f"synthetic_{feature}_percentile_note": "fora dos limites Tukey" if is_outside else "dentro dos limites Tukey",
        })
        pair_distant |= pair_distance_iqr > pair_iqr_threshold
        synthetic_outside |= is_outside
        original_outside_same_direction &= original_outside and same_direction

    row["pair_status"] = "distant" if pair_distant else "close"
    row["reference_status"] = "outside" if synthetic_outside else "within"
    if pair_distant and synthetic_outside:
        status, decision_reason = "reject", "distante do par e fora da referência real"
    elif pair_distant:
        status, decision_reason = "manual_review", "distante do par, apesar de ainda plausível na referência"
    elif synthetic_outside and original_outside_same_direction:
        status, decision_reason = "approved_with_observation", "caso raro preservado de modo compatível com a original"
    elif synthetic_outside:
        status, decision_reason = "manual_review", "fora da referência sem o mesmo padrão raro na original"
    else:
        status, decision_reason = "approved", "próxima do par e dentro da referência real"
    row.update({"status": status, "decision_reason": decision_reason})
    return row
