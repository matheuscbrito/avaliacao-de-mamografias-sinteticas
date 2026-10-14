"""Manifesto, leitura dos pares, alinhamento de intensidade e conjuntos de controle.

Fatos medidos no gen-fid que motivam este módulo:
  * real: float32 sobre a grade 1/4095 (12 bits); sintético: float contínuo;
  * sintético ≈ 0,38·real + 0,34 (afim, linear a 2,7 % do range) — o modelo
    devolve a imagem com faixa de intensidade comprimida;
  * não existe imagem de baixa resolução salva: o controle bicúbico é
    reconstruído a partir do real com fator `k` ainda não confirmado.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.transform import resize

_NAME = re.compile(r"^(?P<batch>\d+)_(?P<abn>Calc|Mass)-(?P<split>Test|Training)_"
                   r"(?P<pid>P_\d+)_(?P<lat>LEFT|RIGHT)_(?P<view>CC|MLO)_(?P<roi>\d+)_roi\.tiff$")


def build_manifest(data_dir: Path) -> pd.DataFrame:
    """Pareia por nome (`X_roi.tiff` ↔ `X_roi_generate.tiff`) e materializa em DataFrame.

    Falha alto se algum real não tiver sintético (ou vice-versa) ou se um nome
    fugir da convenção CBIS-DDSM — pareamento silencioso por `sorted()` de dois
    diretórios é exatamente o erro que destrói o resultado sem avisar.
    """
    data_dir = Path(data_dir)
    reals = sorted(data_dir.glob("*_roi.tiff"))
    syns = {p.name for p in data_dir.glob("*_roi_generate.tiff")}
    rows, orphans = [], []
    for r in reals:
        m = _NAME.match(r.name)
        if not m:
            raise ValueError(f"nome fora da convenção esperada: {r.name}")
        s = r.name.replace("_roi.tiff", "_roi_generate.tiff")
        if s not in syns:
            orphans.append(r.name)
            continue
        syns.discard(s)
        g = m.groupdict()
        rows.append({
            "pair_id": r.name[:-len("_roi.tiff")],
            "real_path": str(r), "syn_path": str(r.with_name(s)),
            "batch": int(g["batch"]), "abnormality": g["abn"], "split": g["split"],
            "patient_id": g["pid"], "image_id": f"{g['pid']}_{g['lat']}_{g['view']}",
            "laterality": g["lat"], "view": g["view"], "roi_index": int(g["roi"]),
        })
    if orphans or syns:
        raise ValueError(f"pareamento incompleto: {len(orphans)} reais sem sintético "
                         f"{orphans[:5]}, {len(syns)} sintéticos sem real {sorted(syns)[:5]}")
    return pd.DataFrame(rows)


def read_tiff(path: str | Path) -> np.ndarray:
    a = tifffile.imread(str(path))
    if a.ndim != 2:
        raise ValueError(f"{path}: esperado 2D, veio {a.shape}")
    return np.asarray(a, dtype=np.float64)


def load_pair(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    return read_tiff(row["real_path"]), read_tiff(row["syn_path"])


# ---------------------------------------------------------------- intensidade

def affine_fit_ls(real: np.ndarray, syn: np.ndarray) -> tuple[float, float]:
    """Mínimos quadrados de `syn ≈ a·real + b`: descreve o que o modelo fez à escala.

    É o número a **reportar** (medido no gen-fid: a ≈ 0,38 — o modelo devolve a
    imagem com a faixa dinâmica comprimida a ~40 %). Não é o que se deve usar
    para normalizar — ver `affine_align`.
    """
    a, b = np.polyfit(real.ravel(), syn.ravel(), 1)
    return float(a), float(b)


def affine_align(real: np.ndarray, syn: np.ndarray, method: str = "moment") -> tuple[np.ndarray, dict]:
    """Traz o sintético para a escala do real. `method='moment'` casa média e desvio.

    **Por que não mínimos quadrados.** A regressão de `syn` em `real` é atenuante:
    o ganho é `cov/var(real)`, então desfazê-lo dá `var(syn_al) = var(real)/corr²`.
    A inflação depende da correlação do par — medida: Pearson(corr, inflação) =
    −0,87. Como os batches diferem em correlação média, normalizar por LS faria a
    comparação entre conjuntos medir qualidade de pareamento disfarçada de
    "excesso de potência". O casamento de momentos (`a = sd(syn)/sd(real)`) é
    independente da correlação e preserva a razão de variâncias.

    Ambos os ganhos vão para o QC: o de LS descreve o modelo, o de momentos
    normaliza a análise.
    """
    a_ls, b_ls = affine_fit_ls(real, syn)
    if method == "moment":
        a = float(syn.std() / real.std())
        b = float(syn.mean() - a * real.mean())
    elif method == "ls":
        a, b = a_ls, b_ls
    else:
        raise ValueError(f"método de alinhamento desconhecido: {method!r} (use 'moment' ou 'ls')")
    if abs(a) < 1e-6:
        raise ValueError(f"ganho de alinhamento degenerado (a={a}); sintético constante?")
    syn_al = (syn - b) / a
    return syn_al, {"affine_a": a, "affine_b": b, "affine_a_ls": a_ls, "affine_b_ls": b_ls,
                    "affine_method_moment": float(method == "moment"),
                    "affine_resid_std": float((syn_al - real).std())}


def requantize(x: np.ndarray, step: float) -> np.ndarray:
    """Arredonda para a grade de quantização do real (1/4095)."""
    return np.rint(x / step) * step


def quantization_step(x: np.ndarray) -> tuple[float, int]:
    """Passo mediano entre valores distintos e número de valores distintos."""
    u = np.unique(x)
    return (float(np.median(np.diff(u))) if u.size > 1 else 0.0), int(u.size)


# ------------------------------------------------------------------ controles

def bicubic_control(real: np.ndarray, k: int) -> np.ndarray:
    """Reduz por `k` com anti-aliasing e volta por interpolação cúbica.

    Emula a entrada de um super-resolvedor clássico. `k` é **inferido**, não
    conhecido — o gerador não salvou a baixa resolução. `skimage.resize(order=3)`
    é spline cúbica (não Keys/`INTER_CUBIC`), mas é passa-baixa equivalente
    para o propósito de controle de suavização.
    """
    n = real.shape[0]
    if n % k:
        raise ValueError(f"lado {n} não é divisível por k={k}")
    low = resize(real, (n // k, n // k), order=3, anti_aliasing=True, preserve_range=True)
    return resize(low, (n, n), order=3, anti_aliasing=False, preserve_range=True)


def butterworth_lowpass(tile: np.ndarray, cutoff_cpx: float, order: int = 4) -> np.ndarray:
    """Passa-baixa Butterworth em frequência, com espelhamento para evitar wraparound.

    |H|² = 1 / (1 + (f/fc)^(2·order)). Com fc = 0,125 c/px e ordem 4, a grade de
    4 px (0,25 c/px) fica atenuada a 0,4 % em amplitude; λ = 16 px passa a 99,6 %.
    """
    n0, n1 = tile.shape
    padded = np.pad(np.asarray(tile, dtype=np.float64), ((n0 // 2, n0 // 2), (n1 // 2, n1 // 2)), mode="reflect")
    fy = np.fft.fftfreq(padded.shape[0])[:, None]
    fx = np.fft.fftfreq(padded.shape[1])[None, :]
    fr = np.hypot(fx, fy)
    H = 1.0 / np.sqrt(1.0 + (fr / cutoff_cpx) ** (2 * order))
    out = np.fft.ifft2(np.fft.fft2(padded) * H).real
    return out[n0 // 2:n0 // 2 + n0, n1 // 2:n1 // 2 + n1]


def highpass_residual_std(real: np.ndarray, syn_al: np.ndarray, cutoff_cpx: float, order: int = 4) -> float:
    """Desvio-padrão da parte de alta frequência de `syn_al − real` (acima do corte)."""
    d = syn_al - real
    return float((d - butterworth_lowpass(d, cutoff_cpx, order)).std())


def noise_control(real: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """`real + N(0, σ²)`: controle na direção observada do modelo (excesso de alta frequência)."""
    return real + rng.normal(0.0, sigma, size=real.shape)


def quadrants(real: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Dois quadrantes disjuntos (diagonal) da mesma ROI: piso de ruído da medição."""
    n = real.shape[0]
    if 2 * size > n:
        raise ValueError(f"quadrante {size} não cabe duas vezes em {n}")
    return real[:size, :size].copy(), real[n - size:, n - size:].copy()


def make_sets(real: np.ndarray, syn: np.ndarray, cfg_common: dict, cfg_ctrl: dict,
              noise_sigma: float | None, rng: np.random.Generator) -> tuple[dict[str, np.ndarray], dict]:
    """Todos os conjuntos de um par, na mesma escala de intensidade e grade de quantização.

    Devolve `{nome: tile}` mais a versão passa-baixa `<nome>_lp` de cada um, e o
    dicionário de QC do alinhamento. `noise_sigma=None` omite `real_noise`
    (primeira passada do notebook 00, antes de o σ global existir).
    """
    step = float(cfg_common["quant_step"])
    syn_al, qc = affine_align(real, syn, cfg_common.get("alignment_method", "moment"))
    sets = {"real": real, "syn": requantize(syn_al, step)}
    for k in cfg_ctrl["bicubic_factors"]:
        sets[f"bicubic_{k}"] = requantize(bicubic_control(real, int(k)), step)
    if noise_sigma is not None:
        sets["real_noise"] = requantize(noise_control(real, noise_sigma, rng), step)
    q1, q2 = quadrants(real, int(cfg_ctrl["quadrant_size_px"]))
    sets["real_q1"], sets["real_q2"] = q1, q2
    fc, order = float(cfg_ctrl["lowpass_cutoff_cpx"]), int(cfg_ctrl["lowpass_order"])
    for name in list(sets):
        sets[f"{name}_lp"] = requantize(butterworth_lowpass(sets[name], fc, order), step)
    return sets, qc


def basic_qc(real: np.ndarray, syn: np.ndarray) -> dict:
    """Flags de entrada por par: correlação, zeros, saturação, quantização, variância."""
    step_r, nd_r = quantization_step(real)
    step_s, nd_s = quantization_step(syn)
    return {
        "corr": float(np.corrcoef(real.ravel(), syn.ravel())[0, 1]),
        "zero_frac_real": float((real == 0).mean()),
        "sat_frac_real": float((real >= real.max()).mean()) if real.max() > 0 else 1.0,
        "range_real": float(real.max() - real.min()), "range_syn": float(syn.max() - syn.min()),
        "mean_real": float(real.mean()), "mean_syn": float(syn.mean()),
        "std_real": float(real.std()), "std_syn": float(syn.std()),
        "distinct_real": float(nd_r), "distinct_syn": float(nd_s),
        "value_step_real": step_r, "value_step_syn": step_s,
        "near_constant": float(real.std() < 1e-6 or syn.std() < 1e-6),
    }
