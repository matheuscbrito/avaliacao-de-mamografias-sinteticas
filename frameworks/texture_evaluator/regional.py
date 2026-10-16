"""Mapa por região e checagem do par original × sintética.

**Checagem do par.** O framework pareia os arquivos pelo nome
(`X.tiff` ↔ `X_generate.tiff` ↔ `X_mask.tiff`). Para pegar arquivo trocado,
mede a correlação de Pearson dos pixels do pulmão entre original e sintética:

- no modo **mesma anatomia** (o gerador parte da própria original, como o Dsr,
  condicionado na versão em baixa resolução), pares verdadeiros ficam ≥ 0,80
  e pares trocados quase sempre < 0,25; abaixo de `PAIR_MIN_CORRELATION` o par
  é marcado como suspeito;
- no modo **só máscara** (o gerador recebe só o contorno do pulmão, como o
  Dgen), a sintética é outro pulmão no mesmo contorno: a correlação é baixa por
  construção, o par só é conferido pelo nome e o mapa por região não se aplica.

O modo de cada versão é detectado pela mediana da correlação dos pares.

**Mapa por região.** Mesmo cálculo de `wavelet/src/mapa_regioes.py`, com a
régua única Tukey k=3: blocos de 16 px com ≥ 90% de pulmão recebem o
deslocamento de brilho (LL) e o veredito de textura (energia fina e grossa em
% do mesmo bloco da original). Daí saem a área do pulmão com brilho local
alterado, a área com textura alterada e a correlação espacial da textura.
"""
from __future__ import annotations

import numpy as np

from wavelet.src.mapa_regioes import VERDICTS, block_maps, compare_maps, summarise

BLOCK = 16
MIN_TISSUE = .9
PAIR_MIN_CORRELATION = .5
SAME_ANATOMY = "mesma anatomia"
MASK_ONLY = "só máscara"


def calibrate_maps(valid, ruler: dict, interval: dict, limits: dict) -> dict:
    """Piso das energias por bloco (P5 dos blocos das originais) e tolerância de brilho local na escala da régua."""
    maps = [block_maps(c.original, c.lung_mask, ruler, BLOCK, "haar", MIN_TISSUE) for c in valid]
    floor = {key: float(np.percentile(np.concatenate([m[key][lung] for m, lung in maps]), 5)) for key in ("fine", "coarse")}
    # a tolerância da densidade (IQR das medianas) está na escala P1–P99; aqui a LL está na escala da régua Tukey
    shift = limits["lung_median"] * (interval["p99"] - interval["p1"]) / (ruler["p99"] - ruler["p1"])
    return {"floor": floor, "bright_shift": float(shift)}


def pair_correlation(original: np.ndarray, synthetic: np.ndarray, mask: np.ndarray) -> float:
    return float(np.corrcoef(original[mask], synthetic[mask])[0, 1])


def regional(original, synthetic, mask, ctx):
    """Leituras por bloco, blocos de pulmão e resumo do mapa."""
    maps, lung = block_maps(original, mask, ctx.ruler, BLOCK, "haar", MIN_TISSUE)
    generated, _ = block_maps(synthetic, mask, ctx.ruler, BLOCK, "haar", MIN_TISSUE)
    readings = compare_maps(maps, generated, lung, ctx.maps["floor"])
    return readings, lung, summarise(readings, maps, generated, lung, ctx.maps["bright_shift"])


def regional_columns(stats: dict) -> dict:
    """Colunas do mapa na linha de cada corte."""
    return {
        "map_area_textura_alterada_pct": 100 * (1 - stats["area_textura preservada"]),
        **{f"map_area_{name.replace(' ', '_')}_pct": 100 * stats[f"area_{name}"] for name in VERDICTS},
        "map_area_brilho_local_pct": 100 * (stats["area_mais_claro"] + stats["area_mais_escuro"]),
        "map_area_mais_claro_pct": 100 * stats["area_mais_claro"],
        "map_area_mais_escuro_pct": 100 * stats["area_mais_escuro"],
        "map_corr_textura": stats["correlacao_textura"],
        "map_n_blocos": stats["n_blocos"],
    }


def apply_pair_check(rows: list[dict]) -> str:
    """Detecta o modo da versão e marca pares suspeitos (só no modo mesma anatomia).

    No modo só máscara, as colunas do mapa ficam vazias: sem a mesma anatomia,
    comparar bloco com bloco não quer dizer nada.
    """
    mode = SAME_ANATOMY if np.median([float(row["pair_correlation"]) for row in rows]) >= PAIR_MIN_CORRELATION else MASK_ONLY
    for row in rows:
        row["pair_mode"] = mode
        if mode == MASK_ONLY:
            row["pair_status"] = "não verificável"
            for key in [key for key in row if key.startswith("map_")]:
                row[key] = float("nan")
        elif float(row["pair_correlation"]) < PAIR_MIN_CORRELATION:
            row["pair_status"] = "suspeito"
            row["final_status"] = "rejeitar"
            row["final_reasons"] = f"par suspeito: correlação {float(row['pair_correlation']):.2f} com a original (arquivo trocado?); {row['final_reasons']}"
        else:
            row["pair_status"] = "ok"
    return mode
