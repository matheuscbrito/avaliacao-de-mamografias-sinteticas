"""Eixo de densidade: o pulmão está claro ou escuro demais?

As features de textura (wavelet, GLCM, NEx) foram feitas para ignorar o nível
médio de intensidade, então não percebem um pulmão que virou "tecido mole".
Este eixo olha só para o histograma dos pixels dentro da máscara:

1. junta os pixels pulmonares de todas as originais aceitas e pega P1 e P99 —
   a faixa normal de um pulmão real;
2. para cada imagem, mede quanto do pulmão cai acima do P99 (claro demais),
   abaixo do P1 (escuro demais) e onde fica a mediana dentro da faixa;
3. compara a sintética com a **sua própria original**, não com um valor fixo:
   há cortes reais legitimamente escuros, e a sintética que os acompanha está
   certa. Os limites vêm da variação entre as originais (Tukey e IQR), a mesma
   régua usada no resto do framework.
"""
from __future__ import annotations

import numpy as np

DENSITY_FEATURES = ("lung_median", "frac_bright", "frac_dark")
# do mais grave ao mais leve: pixels saindo da faixa real > pulmão inteiro deslocado
DENSITY_STATUSES = ("too_bright", "too_dark", "brighter", "darker", "ok")


def reference_interval(images_and_masks: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    """P1, mediana e P99 dos pixels pulmonares de todas as originais juntas."""
    pixels = np.concatenate([image[mask] for image, mask in images_and_masks])
    p1, p50, p99 = (float(value) for value in np.percentile(pixels, (1, 50, 99)))
    return {"p1": p1, "p50": p50, "p99": p99}


def density_features(image: np.ndarray, mask: np.ndarray, interval: dict[str, float]) -> dict[str, float]:
    """Mediana e frações claras/escuras do pulmão, na escala da faixa de referência.

    `lung_median` é a posição da mediana dentro da faixa P1–P99 (0 = P1, 1 = P99),
    para que o número seja legível sem conhecer a unidade da imagem.
    """
    lung = image[mask]
    width = interval["p99"] - interval["p1"]
    return {
        "lung_median": float((np.median(lung) - interval["p1"]) / width),
        "frac_bright": float(np.mean(lung > interval["p99"])),
        "frac_dark": float(np.mean(lung < interval["p1"])),
    }


def density_limits(original_features: list[dict[str, float]]) -> dict[str, float]:
    """Tolerâncias tiradas da variação entre as originais.

    - `frac_bright`/`frac_dark`: limite superior de Tukey (Q3 + 1,5·IQR) da
      fração entre as originais — quanto de pulmão fora da faixa é normal;
    - `lung_median`: IQR das medianas das originais — quanto um pulmão real
      costuma variar de outro.
    """
    limits = {}
    for feature in ("frac_bright", "frac_dark"):
        q1, q3 = np.percentile([item[feature] for item in original_features], (25, 75))
        limits[feature] = float(q3 + 1.5 * (q3 - q1))
    q1, q3 = np.percentile([item["lung_median"] for item in original_features], (25, 75))
    limits["lung_median"] = float(q3 - q1)
    return limits


def density_shift(synthetic: dict[str, float], original: dict[str, float], limits: dict[str, float]) -> dict[str, float]:
    """Diferenças sintética − original do par; a mediana vai em unidades de IQR."""
    return {
        "excess_bright": synthetic["frac_bright"] - original["frac_bright"],
        "excess_dark": synthetic["frac_dark"] - original["frac_dark"],
        "median_shift_iqr": (synthetic["lung_median"] - original["lung_median"]) / max(limits["lung_median"], 1e-12),
    }


def density_status(synthetic: dict[str, float], original: dict[str, float], limits: dict[str, float], median_threshold: float = 1.0) -> str:
    """Classifica a sintética contra a própria original.

    `too_bright`/`too_dark`: a sintética põe mais pulmão fora da faixa real do que
    a original, além da tolerância — sinal de saturação ou de tecido trocado.
    `brighter`/`darker`: nada saiu da faixa, mas a mediana andou mais de
    `median_threshold` IQR — o pulmão inteiro foi deslocado.
    """
    shift = density_shift(synthetic, original, limits)
    bright = shift["excess_bright"] - limits["frac_bright"]
    dark = shift["excess_dark"] - limits["frac_dark"]
    if max(bright, dark) > 0:
        return "too_bright" if bright >= dark else "too_dark"
    if shift["median_shift_iqr"] > median_threshold:
        return "brighter"
    if shift["median_shift_iqr"] < -median_threshold:
        return "darker"
    return "ok"
