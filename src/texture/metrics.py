"""Métricas de textura por tile e métricas pareadas contra o real.

Todas operam sobre tiles já na mesma escala de intensidade (alinhamento afim)
e na mesma grade de quantização (ver `io.make_sets`). Duas famílias:

* **por tile** — um escalar por (par, conjunto); entram no arcabouço de AUC
  real-vs-controle do notebook 03;
* **pareadas** — um escalar por (par, conjunto) que compara o conjunto com o
  real do mesmo par (SSIM, PSNR, JS de LBP, recall de top-hat); para `real`
  são degeneradas e no notebook 03 são comparadas com o piso `real_q1 × real_q2`.

O espectro de potência é estatística de 2ª ordem (autocorrelação) e descarta
a fase; tudo aqui mede algo que a fase carrega.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.ndimage import label, laplace, maximum_filter, sobel, uniform_filter
from scipy.spatial.distance import jensenshannon
from scipy.stats import kurtosis
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.morphology import disk, white_tophat

# ------------------------------------------------------------------ gradiente

def gradient_magnitude(tile: np.ndarray) -> np.ndarray:
    return np.hypot(sobel(tile, axis=0), sobel(tile, axis=1))


def gradient_stats(tile: np.ndarray, threshold: float) -> dict:
    """Distribuição de |∇I| (Sobel) + variância do Laplaciano + Tenengrad.

    Tecido real tem cauda pesada (muitos pixels lisos, poucas transições
    fortes). `threshold` é fixo e global — o mesmo para todos os conjuntos.
    """
    g = gradient_magnitude(tile)
    return {
        "grad_median": float(np.median(g)), "grad_p95": float(np.percentile(g, 95)),
        "grad_kurtosis": float(kurtosis(g.ravel(), fisher=True)),
        "grad_frac_above_thr": float((g > threshold).mean()),
        "laplacian_var": float(laplace(tile).var()),
        "tenengrad": float((g ** 2).mean()),
    }


# ----------------------------------------------------------------------- GLCM

def global_quant_reference(tiles: Sequence[np.ndarray], clip_percentiles: Sequence[float],
                           max_samples_per_tile: int = 50_000, seed: int = 0) -> tuple[float, float]:
    """Limites globais de quantização: percentis sobre amostras agrupadas do conjunto REAL.

    Bins idênticos para todos os conjuntos — quantizar cada imagem pelo próprio
    min/max transformaria diferença de faixa dinâmica em "textura".
    """
    rng = np.random.default_rng(seed)
    pooled = []
    for t in tiles:
        v = np.asarray(t).ravel()
        pooled.append(rng.choice(v, size=min(v.size, max_samples_per_tile), replace=False))
    pooled = np.concatenate(pooled)
    lo, hi = (float(np.percentile(pooled, p)) for p in clip_percentiles)
    if hi <= lo:
        raise ValueError(f"limites de quantização degenerados: lo={lo}, hi={hi}")
    return lo, hi


def quantize(tile: np.ndarray, lo: float, hi: float, levels: int) -> np.ndarray:
    norm = np.clip((tile - lo) / (hi - lo), 0.0, 1.0)
    return np.rint(norm * (levels - 1)).astype(np.uint8)


def glcm_features(q: np.ndarray, cfg: dict) -> dict:
    """Features por distância × ângulo, mais média e desvio angular (anisotropia).

    **Cuidado com o rótulo da distância.** `graycomatrix` arredonda o
    deslocamento para pixels inteiros, então nas diagonais a separação real não é
    `d`: para d=1 a 45° o offset é (−1,+1), ou seja 1,41 px; para d=2 a 45° o
    arredondamento dá o **mesmo** (−1,+1), e as features de `d2_a45`/`d2_a135`
    saem idênticas às de `d1`. `stats.drop_duplicate_metrics` remove essas
    duplicatas antes da correção de FDR. A convenção (d, ângulo) é mantida no
    nome porque é a da literatura de Haralick.
    """
    distances = [int(d) for d in cfg["distances_px"]]
    angles_deg = [float(a) for a in cfg["angles_deg"]]
    glcm = graycomatrix(q, distances=distances, angles=[np.deg2rad(a) for a in angles_deg],
                        levels=int(cfg["n_levels"]), symmetric=True, normed=True)
    out = {}
    for prop in cfg["props"]:
        vals = graycoprops(glcm, prop)                       # (n_dist, n_angle)
        for di, d in enumerate(distances):
            for ai, a in enumerate(angles_deg):
                out[f"glcm_{prop}_d{d}_a{a:g}"] = float(vals[di, ai])
            out[f"glcm_{prop}_d{d}_angmean"] = float(vals[di].mean())
            out[f"glcm_{prop}_d{d}_angstd"] = float(vals[di].std())
    # entropia da GLCM (não está em graycoprops)
    for di, d in enumerate(distances):
        ent = []
        for ai in range(len(angles_deg)):
            p = glcm[:, :, di, ai]
            p = p[p > 0]
            ent.append(float(-(p * np.log2(p)).sum()))
        out[f"glcm_entropy_d{d}_angmean"] = float(np.mean(ent))
        out[f"glcm_entropy_d{d}_angstd"] = float(np.std(ent))
    return out


# ------------------------------------------------------------------------ LBP

def lbp_histogram(tile: np.ndarray, P: int, R: float, method: str = "uniform") -> np.ndarray:
    """Histograma normalizado dos códigos LBP (`uniform`: P+2 bins)."""
    codes = local_binary_pattern(tile, P, R, method=method)
    n_bins = P + 2 if method == "uniform" else 2 ** P
    h, _ = np.histogram(codes, bins=n_bins, range=(0, n_bins))
    return h / h.sum()


def lbp_scalars(hist: np.ndarray, tag: str) -> dict:
    """Escalares por tile a partir do histograma: entropia e fração não-uniforme (último bin)."""
    p = hist[hist > 0]
    return {f"lbp_{tag}_entropy": float(-(p * np.log2(p)).sum()),
            f"lbp_{tag}_nonuniform_frac": float(hist[-1]),
            f"lbp_{tag}_flat_frac": float(hist[0] + hist[-2])}   # tudo-0 e tudo-1: regiões planas


def js_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """Distância de Jensen-Shannon (base 2, em [0, 1]); 0 = histogramas idênticos."""
    return float(jensenshannon(h1, h2, base=2))


# --------------------------------------------------------------- lacunaridade

def lacunarity(tile: np.ndarray, box_sizes: Sequence[int]) -> dict:
    """Lacunaridade por gliding box em escala de cinza: Λ(r) = 1 + var(M_r)/mean(M_r)².

    `M_r` é a massa (soma de intensidade) da janela r×r deslizante. Mede a
    heterogeneidade dos "vazios" por escala — a propriedade que β não captura.
    Intensidade deslocada para ≥ 0 pelo mínimo do tile.
    """
    x = tile - tile.min()
    out = {}
    for r in box_sizes:
        r = int(r)
        m = uniform_filter(x, size=r, mode="reflect")[r // 2:-(r // 2) or None, r // 2:-(r // 2) or None]
        mean = m.mean()
        out[f"lacunarity_{r}px"] = float(1.0 + m.var() / mean ** 2) if mean > 0 else np.nan
    return out


# ----------------------------------------------------------------- top-hat

def tophat_objects(tile: np.ndarray, radius: int, threshold: float | None, k_mad: float,
                   min_area: int) -> tuple[np.ndarray, float]:
    """Máscara de objetos pequenos e brilhantes (white top-hat > limiar) e o limiar usado.

    Se `threshold` for None, deriva-o do próprio tile (mediana + k·MAD do top-hat);
    caso contrário aplica o valor dado (o do real do mesmo par, para todos os conjuntos).
    """
    th = white_tophat(tile, disk(radius))
    if threshold is None:
        med = np.median(th)
        mad = np.median(np.abs(th - med)) * 1.4826
        threshold = float(med + k_mad * mad)
    mask = th > threshold
    lab, n = label(mask)
    if n:
        sizes = np.bincount(lab.ravel())[1:]
        keep = np.isin(lab, np.flatnonzero(sizes >= min_area) + 1)
        mask = keep
    return mask, threshold


def tophat_stats(tile: np.ndarray, real_mask: np.ndarray, threshold: float, cfg: dict) -> dict:
    """Contagem de objetos no tile e recall dos objetos do real (estrutural, não textura)."""
    mask, _ = tophat_objects(tile, int(cfg["radius_px"]), threshold, float(cfg["k_mad"]), int(cfg["min_area_px"]))
    lab, n = label(mask)
    lab_r, n_r = label(real_mask)
    if n_r == 0:
        recall = np.nan
    else:
        near = maximum_filter(mask.astype(np.uint8), size=2 * int(cfg["match_radius_px"]) + 1) > 0
        hits = sum(1 for i in range(1, n_r + 1) if near[lab_r == i].any())
        recall = hits / n_r
    return {"tophat_n_objects": float(n), "tophat_area_frac": float(mask.mean()),
            "tophat_recall_vs_real": float(recall)}


# ------------------------------------------------------------------ pareadas

def paired_fidelity(real: np.ndarray, other: np.ndarray, win: int = 7) -> dict:
    """SSIM / PSNR contra o real do par, `data_range` do real (padrão de `lung_eda.py`).

    Em super-resolução ambos premiam borrão: um modelo que devolve a média das
    texturas plausíveis maximiza PSNR. Servem como referência de fidelidade
    estrutural, não como evidência de textura.
    """
    dr = float(real.max() - real.min())
    mse = float(np.mean((real - other) ** 2))
    # `other is real` (o conjunto `real` comparado consigo) dá MSE 0 e PSNR
    # infinito por definição. Devolver inf explicitamente evita o RuntimeWarning
    # do skimage; `stats` converte inf em NaN e a métrica pareada fica fora da
    # tabela de decisão de qualquer forma (ver `stats.is_paired_metric`).
    psnr = float("inf") if mse == 0 else float(peak_signal_noise_ratio(real, other, data_range=dr))
    return {"ssim_vs_real": float(structural_similarity(real, other, data_range=dr, win_size=win)),
            "psnr_vs_real": psnr,
            "mae_vs_real": float(np.abs(real - other).mean())}
