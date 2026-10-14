"""Execução do pipeline: inventário/QC (00), espectro (01) e métricas (02).

Os notebooks chamam estas funções e cuidam de figuras e narrativa; o cálculo
vive aqui para poder rodar fora do Jupyter (`python -m texture.pipeline ...`).

Ordem obrigatória: `run_inventory` deriva as constantes globais (σ do controle
de ruído, limiar de gradiente, limites de quantização da GLCM) que `run_spectrum`
e `run_metrics` consomem. Rodar fora de ordem falha alto.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import io as tio
from . import metrics as tmet
from . import spectrum as tspec
from .store import Config, write_long

# conjuntos que existem depois de `io.make_sets` (com σ definido)
ALL_SETS = ["real", "syn", "bicubic_2", "bicubic_4", "real_noise", "real_q1", "real_q2"]
ALL_SETS += [f"{s}_lp" for s in ALL_SETS]
CONTROL_SETS = ["bicubic_2", "bicubic_4", "real_noise"]


def _derived_path(cfg: Config) -> Path:
    return cfg.out / "derived_constants.json"


def load_derived(cfg: Config) -> dict:
    p = _derived_path(cfg)
    if not p.exists():
        raise FileNotFoundError(f"{p} não existe — rode run_inventory() (notebook 00) antes.")
    return json.loads(p.read_text())


# ------------------------------------------------------------------ notebook 00

def run_inventory(cfg: Config, limit: int | None = None, verbose: bool = True) -> pd.DataFrame:
    """Manifesto + QC por par + constantes globais derivadas.

    Nenhuma ROI é excluída: tudo vira flag. As constantes derivadas (σ do ruído,
    limiar de gradiente, limites da GLCM) saem do conjunto REAL e são gravadas
    para que os três conjuntos usem exatamente os mesmos valores.
    """
    man = tio.build_manifest(cfg.paths["data_dir"])
    if limit:
        man = man.head(limit)
    man.to_csv(cfg.out / "manifest.csv", index=False)

    common, ctrl, psd_cfg = cfg["common"], cfg["controls"], cfg["psd"]
    qc_hash = cfg.hash("common", "alignment", "controls", "psd")
    rows, sigmas, grad_pool, real_tiles = [], [], [], []

    for i, row in man.iterrows():
        real, syn = tio.load_pair(row)
        qc = tio.basic_qc(real, syn)
        syn_al, aff = tio.affine_align(real, syn)
        qc.update(aff)
        for per in psd_cfg["harmonic_periods_px"]:
            qc[f"harmonic_{per}px_real"] = tspec.harmonic_excess(real, int(per))[0]
            qc[f"harmonic_{per}px_syn"] = tspec.harmonic_excess(syn_al, int(per))[0]
        # período 2 px: xadrez de upsampling. Medido em separado porque é
        # artefato de geração e separa os conjuntos sozinho (AUC 1,0).
        qc["nyquist_excess_real"] = tspec.nyquist_excess(real)[0]
        qc["nyquist_excess_syn"] = tspec.nyquist_excess(syn_al)[0]
        sigmas.append(tio.highpass_residual_std(real, syn_al, float(ctrl["lowpass_cutoff_cpx"]),
                                                int(ctrl["lowpass_order"])))
        qc["highpass_residual_std"] = sigmas[-1]
        for flag, value in qc.items():
            rows.append({"pair_id": row["pair_id"], "flag": flag, "value": float(value),
                         "config_hash": qc_hash})
        g = tmet.gradient_magnitude(real)
        grad_pool.append(np.random.default_rng(i).choice(g.ravel(), size=20_000, replace=False))
        real_tiles.append(real)
        if verbose and (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(man)} pares", flush=True)

    write_long(cfg.out / "qc.parquet", rows, ["pair_id", "flag", "config_hash"])

    lo, hi = tmet.global_quant_reference(real_tiles, cfg["glcm"]["clip_percentiles"], seed=0)
    derived = {
        "noise_sigma": float(np.median(sigmas)),
        "gradient_threshold": float(np.percentile(np.concatenate(grad_pool),
                                                  float(cfg["gradient"]["threshold_percentile_real"]))),
        "glcm_quant_lo": lo, "glcm_quant_hi": hi,
        "n_pairs": int(len(man)), "config_hash": qc_hash,
    }
    _derived_path(cfg).write_text(json.dumps(derived, indent=2))
    cfg.sidecar("sidecar_00_inventory", ["common", "alignment", "controls", "psd", "glcm", "gradient"],
                derived=derived, physical_units=cfg["common"]["physical_units"],
                bicubic_origin="inferido: nenhuma imagem de baixa resolução foi salva pelo gerador",
                input_is_real_roi="não confirmado")
    if verbose:
        qc_df = pd.DataFrame(rows).pivot_table(index="pair_id", columns="flag", values="value")
        print(f"\n{len(man)} pares | σ ruído {derived['noise_sigma']:.5f} | "
              f"limiar grad {derived['gradient_threshold']:.5f} | GLCM [{lo:.4f}, {hi:.4f}]")
        print(f"corr mediana {qc_df['corr'].median():.3f} | "
              f"afim a mediana {qc_df['affine_a'].median():.3f} | "
              f"harm 4px syn mediana {qc_df['harmonic_4px_syn'].median():.2f} decadas")
    return man


# ------------------------------------------------------------------ notebook 01

def run_spectrum(cfg: Config, limit: int | None = None, verbose: bool = True) -> None:
    """Perfis radiais + escalares espectrais para todos os conjuntos."""
    man = pd.read_csv(cfg.out / "manifest.csv")
    if limit:
        man = man.head(limit)
    derived = load_derived(cfg)
    common, ctrl, psd_cfg = cfg["common"], cfg["controls"], cfg["psd"]
    h = cfg.hash("common", "alignment", "controls", "psd")
    rng = np.random.default_rng(int(ctrl["noise_seed"]))

    prof_rows, met_rows = [], []
    for i, row in man.iterrows():
        real, syn = tio.load_pair(row)
        sets, _ = tio.make_sets(real, syn, common, ctrl, derived["noise_sigma"], rng)
        for name, tile in sets.items():
            bins = psd_cfg["fit_bins_quadrant"] if tile.shape[0] < int(common["tile_size_px"]) else psd_cfg["fit_bins"]
            scal, rad = tspec.spectral_scalars(tile, psd_cfg, bins)
            for k, v in scal.items():
                met_rows.append({"pair_id": row["pair_id"], "set": name, "metric": k,
                                 "value": float(v), "config_hash": h})
            for b, f, p in zip(rad["bin_index"], rad["freq_cpx"], rad["power"]):
                prof_rows.append({"pair_id": row["pair_id"], "set": name, "radial_bin": int(b),
                                  "freq_cpx": float(f), "power": float(p), "config_hash": h})
        if verbose and (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(man)} pares", flush=True)

    write_long(cfg.out / "profiles.parquet", prof_rows, ["pair_id", "set", "radial_bin", "config_hash"])
    write_long(cfg.out / "metrics.parquet", met_rows, ["pair_id", "set", "metric", "config_hash"])
    cfg.sidecar("sidecar_01_spectrum", ["common", "alignment", "controls", "psd"],
                derived=derived, units="ciclos/pixel", physical_units=False)
    if verbose:
        print(f"\n{len(prof_rows)} linhas de perfil, {len(met_rows)} escalares espectrais")


# ------------------------------------------------------------------ notebook 02

def run_metrics(cfg: Config, limit: int | None = None, verbose: bool = True) -> None:
    """Gradiente, GLCM, LBP, lacunaridade, top-hat e pareadas para todos os conjuntos."""
    man = pd.read_csv(cfg.out / "manifest.csv")
    if limit:
        man = man.head(limit)
    derived = load_derived(cfg)
    common, ctrl = cfg["common"], cfg["controls"]
    glcm_cfg, lbp_cfg = cfg["glcm"], cfg["lbp"]
    lac_cfg, th_cfg, paired_cfg = cfg["lacunarity"], cfg["tophat"], cfg["paired"]
    h = cfg.hash("common", "alignment", "controls", "gradient", "glcm", "lbp", "lacunarity",
                 "tophat", "paired")
    rng = np.random.default_rng(int(ctrl["noise_seed"]))
    lo, hi = derived["glcm_quant_lo"], derived["glcm_quant_hi"]
    thr = derived["gradient_threshold"]

    rows = []
    for i, row in man.iterrows():
        real, syn = tio.load_pair(row)
        sets, _ = tio.make_sets(real, syn, common, ctrl, derived["noise_sigma"], rng)
        # referências do REAL do mesmo par, aplicadas a todos os conjuntos
        real_mask, real_thr = tmet.tophat_objects(sets["real"], int(th_cfg["radius_px"]), None,
                                                  float(th_cfg["k_mad"]), int(th_cfg["min_area_px"]))
        lbp_ref = {tag: tmet.lbp_histogram(sets["real"], P, R, lbp_cfg["method"])
                   for (P, R) in lbp_cfg["configs"] for tag in [f"P{P}R{R}"]}
        real_lp_mask, real_lp_thr = tmet.tophat_objects(sets["real_lp"], int(th_cfg["radius_px"]), None,
                                                        float(th_cfg["k_mad"]), int(th_cfg["min_area_px"]))
        lbp_ref_lp = {tag: tmet.lbp_histogram(sets["real_lp"], P, R, lbp_cfg["method"])
                      for (P, R) in lbp_cfg["configs"] for tag in [f"P{P}R{R}"]}

        for name, tile in sets.items():
            is_lp = name.endswith("_lp")
            is_quadrant = tile.shape[0] < int(common["tile_size_px"])
            out = {}
            out.update(tmet.gradient_stats(tile, thr))
            out.update(tmet.glcm_features(tmet.quantize(tile, lo, hi, int(glcm_cfg["n_levels"])), glcm_cfg))
            out.update(tmet.lacunarity(tile, lac_cfg["box_sizes_px"]))
            for (P, R) in lbp_cfg["configs"]:
                tag = f"P{P}R{R}"
                hist = tmet.lbp_histogram(tile, P, R, lbp_cfg["method"])
                out.update(tmet.lbp_scalars(hist, tag))
                if not is_quadrant:
                    ref = (lbp_ref_lp if is_lp else lbp_ref)[tag]
                    out[f"lbp_{tag}_js_vs_real"] = tmet.js_distance(ref, hist)
            if not is_quadrant:
                base = sets["real_lp"] if is_lp else sets["real"]
                mask, mthr = (real_lp_mask, real_lp_thr) if is_lp else (real_mask, real_thr)
                out.update(tmet.tophat_stats(tile, mask, mthr, th_cfg))
                out.update(tmet.paired_fidelity(base, tile, int(paired_cfg["ssim_win"])))
            for k, v in out.items():
                rows.append({"pair_id": row["pair_id"], "set": name, "metric": k,
                             "value": float(v), "config_hash": h})
        if verbose and (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(man)} pares", flush=True)

    write_long(cfg.out / "metrics.parquet", rows, ["pair_id", "set", "metric", "config_hash"])
    cfg.sidecar("sidecar_02_metrics",
                ["common", "alignment", "controls", "gradient", "glcm", "lbp", "lacunarity",
                 "tophat", "paired"],
                derived=derived,
                glcm_quantization={"scheme": "global", "lo": lo, "hi": hi,
                                   "n_levels": glcm_cfg["n_levels"], "source": "conjunto real"})
    if verbose:
        print(f"\n{len(rows)} escalares de textura gravados")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    c = Config()
    if stage in ("00", "inventory", "all"):
        run_inventory(c, lim)
    if stage in ("01", "spectrum", "all"):
        run_spectrum(c, lim)
    if stage in ("02", "metrics", "all"):
        run_metrics(c, lim)
