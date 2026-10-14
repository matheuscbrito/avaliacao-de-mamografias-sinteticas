"""Estatística da decisão: AUC, viés pareado, dispersão, piso de ruído, FDR.

Duas regras que atravessam o módulo:

* **Agrupamento por paciente.** São 225 pares para 202 pacientes (6 imagens com
  mais de uma ROI). Bootstrap e IC reamostram *pacientes*, não pares — ROIs da
  mesma paciente não são observações independentes.
* **Viés e dispersão nunca colapsam num número.** Um modelo pode acertar a
  mediana e colapsar a diversidade; o teste de viés não vê isso.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, kruskal, mannwhitneyu, wilcoxon


def auc_mannwhitney(a: Sequence[float], b: Sequence[float]) -> float:
    """AUC de um classificador de uma feature só (estatística U normalizada).

    0,5 = indistinguível, 1,0 = separação perfeita. Devolvida como
    `max(auc, 1-auc)`: a direção da diferença é lida no viés, não aqui, e uma
    métrica que separa invertida separa igual.
    """
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return np.nan
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    auc = u / (a.size * b.size)
    return float(max(auc, 1.0 - auc))


def _grouped_resample(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Índices de uma reamostragem com reposição *por grupo* (cluster bootstrap)."""
    uniq = np.unique(groups)
    picked = rng.choice(uniq, size=uniq.size, replace=True)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    return np.concatenate([idx[g] for g in picked])


def bootstrap_ci(values: np.ndarray, groups: np.ndarray, statistic, n_boot: int = 1000,
                 seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """(estimativa, IC inferior, IC superior) por bootstrap agrupado por paciente."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    ok = np.isfinite(values)
    values, groups = values[ok], groups[ok]
    if values.size < 3:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    est = float(statistic(values))
    boot = np.array([statistic(values[_grouped_resample(groups, rng)]) for _ in range(n_boot)])
    boot = boot[np.isfinite(boot)]
    if boot.size < 10:
        return est, np.nan, np.nan
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return est, float(lo), float(hi)


def geometric_mean_ratio_ci(ratios: np.ndarray, groups: np.ndarray, n_boot: int = 1000,
                            seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Média geométrica por bin (coluna) de uma matriz de razões, com IC bootstrap.

    Razão é grandeza multiplicativa: a média aritmética é enviesada para cima.
    Média no espaço log e volta pela exponencial.
    """
    lr = np.log(np.asarray(ratios, dtype=float))
    groups = np.asarray(groups)
    est = np.exp(np.nanmean(lr, axis=0))
    rng = np.random.default_rng(seed)
    boot = np.empty((n_boot, lr.shape[1]))
    for i in range(n_boot):
        boot[i] = np.exp(np.nanmean(lr[_grouped_resample(groups, rng)], axis=0))
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return est, lo, hi


def paired_bias(real: np.ndarray, other: np.ndarray, groups: np.ndarray,
                n_boot: int = 1000, seed: int = 0) -> dict:
    """Wilcoxon pareado de `other − real` + mediana da diferença com IC agrupado."""
    real = np.asarray(real, dtype=float); other = np.asarray(other, dtype=float)
    ok = np.isfinite(real) & np.isfinite(other)
    d = other[ok] - real[ok]
    g = np.asarray(groups)[ok]
    if d.size < 5 or np.allclose(d, 0):
        return {"bias_median": np.nan, "bias_lo": np.nan, "bias_hi": np.nan, "p_value": np.nan, "n": int(d.size)}
    med, lo, hi = bootstrap_ci(d, g, np.median, n_boot, seed)
    p = float(wilcoxon(d, alternative="two-sided", zero_method="zsplit").pvalue)
    return {"bias_median": med, "bias_lo": lo, "bias_hi": hi, "p_value": p, "n": int(d.size)}


def dispersion_ratio(real: np.ndarray, other: np.ndarray, groups: np.ndarray,
                     n_boot: int = 1000, seed: int = 0) -> dict:
    """`sd(other)/sd(real)` com IC bootstrap agrupado. < 1 = colapso de diversidade.

    Sem direção pressuposta: no gen-fid o β do sintético é *mais* disperso que o
    do real, não menos.
    """
    real = np.asarray(real, dtype=float); other = np.asarray(other, dtype=float)
    ok = np.isfinite(real) & np.isfinite(other)
    real, other, g = real[ok], other[ok], np.asarray(groups)[ok]
    if real.size < 5 or real.std() == 0:
        return {"disp_ratio": np.nan, "disp_lo": np.nan, "disp_hi": np.nan}
    rng = np.random.default_rng(seed)
    est = float(other.std() / real.std())
    boot = []
    for _ in range(n_boot):
        i = _grouped_resample(g, rng)
        sr = real[i].std()
        if sr > 0:
            boot.append(other[i].std() / sr)
    if len(boot) < 10:
        return {"disp_ratio": est, "disp_lo": np.nan, "disp_hi": np.nan}
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"disp_ratio": est, "disp_lo": float(lo), "disp_hi": float(hi)}


def noise_floor(q1: np.ndarray, q2: np.ndarray) -> float:
    """Margem de equivalência: p95 do |desvio| entre dois quadrantes da mesma ROI real.

    Desvio menor que isto é indistinguível do ruído de medição da métrica.
    """
    q1 = np.asarray(q1, dtype=float); q2 = np.asarray(q2, dtype=float)
    d = np.abs(q2 - q1)
    d = d[np.isfinite(d)]
    return float(np.percentile(d, 95)) if d.size else np.nan


def fdr(pvalues: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg. NaNs preservados (métricas que não produziram p-valor)."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.sum():
        out[ok] = false_discovery_control(p[ok], method="bh")
    return out


def stratum_heterogeneity(deltas: np.ndarray, strata: np.ndarray) -> float:
    """Kruskal-Wallis dos deltas pareados entre estratos: a consistência é a mesma?

    p pequeno = o desvio sintético−real difere entre batches (ou Calc/Mass, ou
    Test/Training) — a resposta direta a "essa consistência é parecida entre os
    conjuntos".
    """
    deltas = np.asarray(deltas, dtype=float)
    strata = np.asarray(strata)
    ok = np.isfinite(deltas)
    deltas, strata = deltas[ok], strata[ok]
    groups = [deltas[strata == s] for s in np.unique(strata)]
    groups = [g for g in groups if g.size >= 3]
    if len(groups) < 2:
        return np.nan
    try:
        return float(kruskal(*groups).pvalue)
    except ValueError:            # todos os valores idênticos
        return np.nan


def verdict(auc_controls: dict[str, float], auc_syn: float, auc_syn_lp: float,
            auc_floor: float, useful: float = 0.80, pass_band: Sequence[float] = (0.40, 0.60)) -> str:
    """Classifica a métrica segundo a tabela de decisão do plano.

    `auc_controls` = AUC do real contra cada controle (bicubic_*, real_noise).
    Uma métrica só tem direito de opinar sobre o modelo se separa o real de
    algum degradado conhecido E não separa dois quadrantes do mesmo tecido real.
    """
    sensitive = any(np.isfinite(v) and v >= useful for v in auc_controls.values())
    floor_ok = not np.isfinite(auc_floor) or auc_floor < useful
    if not sensitive:
        return "cega"
    if not floor_ok:
        return "instavel"                       # separa tecido real de si mesmo
    lo, hi = pass_band
    syn_sep = np.isfinite(auc_syn) and auc_syn >= useful
    lp_sep = np.isfinite(auc_syn_lp) and auc_syn_lp >= useful
    if syn_sep and lp_sep:
        return "falha_escala_tecido"
    if syn_sep and not lp_sep:
        return "falha_escala_fina"
    if np.isfinite(auc_syn) and lo <= auc_syn <= hi:
        return "modelo_passa"
    return "inconclusiva"


def drop_duplicate_metrics(metrics: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Remove colunas numericamente idênticas a outra, mantendo a primeira.

    **Por que existem duplicatas.** `graycomatrix` arredonda o deslocamento para
    pixels inteiros: `d=2` a 45° vira `round(±2·0,707) = ±1`, exatamente o mesmo
    offset de `d=1` a 45°. Doze features da GLCM saem duplicadas por isso (o
    rótulo `d2_a45` é enganoso: a separação real é 1,41 px, não 2).

    Contá-las como testes distintos infla a correção de Benjamini-Hochberg, que
    pressupõe testes distintos — duplicatas exatas com o mesmo p-valor aumentam
    artificialmente o número de "descobertas". Elas ficam no parquet (o dado é o
    dado) e saem aqui, com o registro do que foi removido.
    """
    cols = list(metrics.columns)
    arrays = {c: metrics[c].to_numpy(dtype=float) for c in cols}
    dropped: dict[str, str] = {}
    keep: list[str] = []
    for c in cols:
        a = arrays[c]
        for k in keep:
            b = arrays[k]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() > 10 and np.allclose(a[m], b[m], rtol=1e-12, atol=1e-15):
                dropped[c] = k
                break
        else:
            keep.append(c)
    if verbose and dropped:
        print(f"{len(dropped)} métricas duplicadas removidas antes da correção de FDR "
              f"(ex.: {next(iter(dropped))} == {dropped[next(iter(dropped))]})")
    return metrics[keep], dropped


#: Métricas que medem **artefato de geração**, não textura de tecido. Sua versão
#: passa-baixa não neutraliza nada — são razões pico/fundo, e o filtro atenua
#: numerador e denominador juntos — então o veredito por banda não se aplica a
#: elas. Entram no relatório sob categoria própria.
ARTIFACT_METRICS = ("nyquist_excess", "harmonic_", "band_frac_corners")


def is_artifact_metric(name: str) -> bool:
    return name.startswith(ARTIFACT_METRICS)


def is_paired_metric(name: str) -> bool:
    """Métricas que comparam um conjunto com o real do mesmo par.

    São **degeneradas no próprio real** por construção (SSIM=1, PSNR=∞, MAE=0,
    JS=0, recall=1), então a AUC "real vs. X" delas é ~1 automaticamente e não
    demonstra sensibilidade nenhuma. Elas saem da tabela de decisão e vão para
    `paired_comparison_table`, que compara os conjuntos **entre si**.
    """
    return name.endswith("_vs_real")


def paired_comparison_table(metrics: pd.DataFrame, manifest: pd.DataFrame,
                            control_sets: Sequence[str], cfg_stats: dict) -> pd.DataFrame:
    """Compara métricas pareadas entre conjuntos (o real é degenerado e fica de fora).

    A leitura aqui é diferente: não é "separa ou não separa", e sim *quão longe do
    real* cada conjunto fica, e se o sintético fica mais longe que uma suavização
    ingênua. Para SSIM/PSNR, um bicúbico melhor que o difusor é o esperado — essas
    métricas premiam borrão — e isso deve ser dito, não escondido.
    """
    sets = [s for s in ["syn", *control_sets] if s in metrics.index.get_level_values("set")]
    rows = []
    for metric in sorted(m for m in metrics.columns if is_paired_metric(m)):
        row = {"metric": metric}
        for s in sets:
            v = metrics.xs(s, level="set")[metric].replace([np.inf, -np.inf], np.nan)
            row[f"{s}_median"] = float(v.median())
        syn = metrics.xs("syn", level="set")[metric].replace([np.inf, -np.inf], np.nan)
        for s in control_sets:
            if s not in sets:
                continue
            ctrl = metrics.xs(s, level="set")[metric].replace([np.inf, -np.inf], np.nan)
            row[f"auc_syn_vs_{s}"] = auc_mannwhitney(syn.to_numpy(), ctrl.to_numpy())
        rows.append(row)
    return pd.DataFrame(rows)


def decision_table(metrics: pd.DataFrame, manifest: pd.DataFrame, control_sets: Sequence[str],
                   cfg_stats: dict) -> pd.DataFrame:
    """Tabela de decisão: uma linha por métrica, ordenada por sensibilidade a controles.

    `metrics` é o parquet longo pivotado por `metrics_wide` (índice pair_id × set).
    Métricas pareadas (`*_vs_real`) são excluídas — ver `is_paired_metric` — e
    duplicatas exatas removidas antes do FDR — ver `drop_duplicate_metrics`.
    """
    metrics, _ = drop_duplicate_metrics(metrics)
    seed, n_boot = int(cfg_stats["seed"]), int(cfg_stats["n_bootstrap"])
    useful, band = float(cfg_stats["auc_useful"]), cfg_stats["auc_pass_band"]
    groups_by_pair = manifest.set_index("pair_id")["patient_id"]

    def series(metric: str, s: str) -> pd.Series:
        if s not in metrics.index.get_level_values("set"):
            return pd.Series(dtype=float)
        return metrics.xs(s, level="set")[metric].replace([np.inf, -np.inf], np.nan)

    rows = []
    for metric in sorted(metrics.columns):
        if is_paired_metric(metric):
            continue
        real = series(metric, "real")
        if real.dropna().empty:
            continue
        pairs = real.index
        groups = groups_by_pair.reindex(pairs).to_numpy()
        aucs = {s: auc_mannwhitney(real.to_numpy(), series(metric, s).reindex(pairs).to_numpy())
                for s in control_sets if s in metrics.index.get_level_values("set")}
        syn = series(metric, "syn").reindex(pairs)
        syn_lp = series(metric, "syn_lp").reindex(pairs)
        real_lp = series(metric, "real_lp").reindex(pairs)
        q1, q2 = series(metric, "real_q1").reindex(pairs), series(metric, "real_q2").reindex(pairs)
        auc_syn = auc_mannwhitney(real.to_numpy(), syn.to_numpy())
        auc_syn_lp = (auc_mannwhitney(real_lp.to_numpy(), syn_lp.to_numpy())
                      if real_lp.notna().any() else np.nan)
        auc_floor = auc_mannwhitney(q1.to_numpy(), q2.to_numpy())
        row = {"metric": metric, "auc_syn": auc_syn, "auc_syn_lp": auc_syn_lp, "auc_floor": auc_floor,
               "noise_floor": noise_floor(q1.to_numpy(), q2.to_numpy())}
        row.update({f"auc_{s}": v for s, v in aucs.items()})
        # `real_noise` foi calibrado para igualar o sintético em potência de alta
        # frequência (razão espectral 2,48 vs 2,47 em λ=4 px). Separar os dois,
        # então, não é ver "quanto" de alta frequência há — é ver a *estrutura*:
        # o ruído é branco, o artefato do modelo é periódico.
        if "real_noise" in metrics.index.get_level_values("set"):
            row["auc_syn_vs_noise"] = auc_mannwhitney(
                syn.to_numpy(), series(metric, "real_noise").reindex(pairs).to_numpy())
        row.update(paired_bias(real.to_numpy(), syn.to_numpy(), groups, n_boot, seed))
        row.update(dispersion_ratio(real.to_numpy(), syn.to_numpy(), groups, n_boot, seed))
        delta = syn.to_numpy() - real.to_numpy()
        for st in cfg_stats["strata"]:
            row[f"p_heterogeneity_{st}"] = stratum_heterogeneity(
                delta, manifest.set_index("pair_id")[st].reindex(pairs).to_numpy())
        row["max_auc_control"] = max([v for v in aucs.values() if np.isfinite(v)], default=np.nan)
        # Tamanho do efeito contra o piso de medição. Com 225 pares, quase toda
        # métrica tem p significativo; o que separa achado de ruído é o desvio
        # comparado ao que dois pedaços do mesmo tecido real já produzem.
        floor_v = row["noise_floor"]
        row["bias_vs_floor"] = (abs(row["bias_median"]) / floor_v
                                if np.isfinite(floor_v) and floor_v > 0 else np.nan)
        if is_artifact_metric(metric):
            # detector de artefato: a leitura por banda não se aplica (a
            # estatística é uma razão e sobrevive ao passa-baixa por construção)
            row["verdict"] = "artefato_de_geracao"
        else:
            row["verdict"] = verdict(aucs, auc_syn, auc_syn_lp, auc_floor, useful, band)
        rows.append(row)

    tab = pd.DataFrame(rows)
    tab["p_value_bh"] = fdr(tab["p_value"], float(cfg_stats["fdr_alpha"]))
    for st in cfg_stats["strata"]:
        tab[f"p_heterogeneity_{st}_bh"] = fdr(tab[f"p_heterogeneity_{st}"], float(cfg_stats["fdr_alpha"]))
    return tab.sort_values("max_auc_control", ascending=False).reset_index(drop=True)
