"""Espectro de potência radial, lei de potência, harmônicos periódicos e rolloff.

Extraído da antiga bancada INbreast (`extracao_metricas_textura.ipynb`,
seções 9, 11 e 12; já removida do repo) e convertido
para **ciclos/pixel**: o pixel spacing das ROIs do gen-fid é desconhecido.
Se `pixel_spacing_mm` for informado, `freq_cpmm = freq_cpx / spacing`.

Pré-processamento espectral (aplicar a TODOS os conjuntos ou a nenhum):
  1. detrend polinomial 2D de 2ª ordem — remove a rampa de espessura, que
     vaza para as primeiras bandas e enviesa β para cima;
  2. janela de Hann periódica com correção de energia ⟨w²⟩ — sem ela o
     vazamento espectral achata a inclinação (medido: −0,56 em β com β=4).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.signal.windows import hann as _hann


def poly_detrend_2d(tile: np.ndarray, order: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Remove um polinômio 2D de grau `order` por mínimos quadrados. Retorna (resíduo, coef)."""
    if tile.ndim != 2:
        raise ValueError(f"esperado tile 2D, veio shape {tile.shape}")
    n0, n1 = tile.shape
    yy, xx = np.mgrid[0:n0, 0:n1].astype(np.float64)
    xx = (xx - (n1 - 1) / 2) / max(n1 - 1, 1) * 2.0
    yy = (yy - (n0 - 1) / 2) / max(n0 - 1, 1) * 2.0
    basis = [(xx ** i) * (yy ** j) for i in range(order + 1) for j in range(order + 1 - i)]
    A = np.stack([b.ravel() for b in basis], axis=1)
    coef, *_ = np.linalg.lstsq(A, tile.ravel(), rcond=None)
    return tile - (A @ coef).reshape(tile.shape), coef


def hann2d(n: int) -> tuple[np.ndarray, float]:
    """Janela de Hann 2D separável (periódica) e seu fator de correção de energia ⟨w²⟩."""
    w1 = _hann(n, sym=False)
    w = np.outer(w1, w1)
    return w, float((w ** 2).mean())


def power_spectrum_2d(tile: np.ndarray, detrend_order: int = 2, use_window: bool = True) -> np.ndarray:
    """PSD 2D em unidades de intensidade² por modo, eixo em ciclos/pixel (não deslocado).

    Normalização `|F|² / (N² ⟨w²⟩)`: a soma sobre a grade recupera a variância do
    resíduo (Parseval), independente da janela.
    """
    if tile.ndim != 2 or tile.shape[0] != tile.shape[1]:
        raise ValueError(f"esperado tile quadrado 2D, veio shape {tile.shape}")
    n = tile.shape[0]
    resid = poly_detrend_2d(np.asarray(tile, dtype=np.float64), detrend_order)[0] if detrend_order >= 0 else tile
    if use_window:
        w, u = hann2d(n)
    else:
        w, u = np.ones((n, n)), 1.0
    F = np.fft.fft2(resid * w)
    return np.abs(F) ** 2 / (n ** 2 * u)


def radial_average(psd: np.ndarray) -> dict:
    """Média radial do PSD, DC excluído, bins até k = N/2 (anéis além só têm cantos).

    A frequência do bin é o centroide de |f| no anel, não o centro nominal — em
    raios pequenos os dois diferem o bastante para enviesar o ajuste log–log.
    """
    n = psd.shape[0]
    f1 = np.fft.fftfreq(n)                       # ciclos/pixel
    fr = np.hypot(f1[None, :], f1[:, None])
    idx = np.rint(fr * n).astype(np.int64)
    nb = int(idx.max()) + 1
    cnt = np.bincount(idx.ravel(), minlength=nb).astype(float)
    psum = np.bincount(idx.ravel(), weights=psd.ravel(), minlength=nb)
    fsum = np.bincount(idx.ravel(), weights=fr.ravel(), minlength=nb)
    k = np.arange(1, n // 2 + 1)
    return {"bin_index": k, "freq_cpx": fsum[k] / cnt[k], "power": psum[k] / cnt[k],
            "n_modes": cnt[k], "f_nyquist_cpx": 0.5}


def fit_power_law(radial: dict, fit_bins: Sequence[int]) -> dict:
    """Regressão log–log de `power` contra `freq` nos bins radiais [lo, hi]. β = −inclinação."""
    lo, hi = int(fit_bins[0]), int(fit_bins[1])
    k, f, p = radial["bin_index"], radial["freq_cpx"], radial["power"]
    m = (k >= lo) & (k <= hi) & (p > 0) & np.isfinite(p)
    npts = int(m.sum())
    if npts < 3:
        raise ValueError(f"apenas {npts} bins positivos em [{lo}, {hi}] — insuficiente para uma reta")
    lf, lp = np.log10(f[m]), np.log10(p[m])
    slope, intercept = np.polyfit(lf, lp, 1)
    pred = slope * lf + intercept
    ss_tot = float(np.sum((lp - lp.mean()) ** 2))
    r2 = 1.0 - float(np.sum((lp - pred) ** 2)) / ss_tot if ss_tot > 0 else np.nan
    return {"beta": float(-slope), "intercept": float(intercept), "r2": float(r2), "n_bins_fit": npts}


def band_fractions(psd: np.ndarray, bands_cpx: Sequence[Sequence[float]]) -> dict:
    """Fração da potência em cada banda de |f| (ciclos/pixel), DC excluído.

    Numerador e denominador são restritos ao **disco isotrópico** |f| ≤ 0,5: os
    cantos da grade (21,5 % dos modos, |f| até 0,707) só existem nas direções
    diagonais, então incluí-los faria a fração depender da geometria da grade e
    da anisotropia do conteúdo, não da textura. É a mesma exclusão que
    `radial_average` aplica a k > N/2, e ela importa: num campo β=3 os cantos
    guardam 0,05 % da potência, mas em conteúdo de alta frequência guardam ~22 %.

    Com bandas que cobrem [0, 0,5] sem buraco, as frações somam exatamente 1.
    `band_frac_corners` registra o que foi excluído, para que a decisão fique
    auditável em vez de silenciosa.
    """
    n = psd.shape[0]
    f1 = np.fft.fftfreq(n)
    fr = np.hypot(f1[None, :], f1[:, None])
    disc = (fr > 0) & (fr <= 0.5)
    total = float(psd[disc].sum())
    out = {}
    for lo, hi in bands_cpx:
        lo, hi = float(lo), float(hi)
        # a banda que termina em Nyquist é fechada à direita, senão a linha/coluna
        # exatamente em |f| = 0,5 não cairia em banda alguma
        upper = (fr <= hi) if hi >= 0.5 else (fr < hi)
        m = disc & (fr >= lo) & upper
        out[f"band_frac_{lo:g}_{hi:g}"] = float(psd[m].sum()) / total if total > 0 else np.nan
    corners = float(psd[fr > 0.5].sum())
    out["band_frac_corners"] = corners / (total + corners) if total + corners > 0 else np.nan
    return out


def _gradient_profile_spectrum(tile: np.ndarray, axis: int) -> np.ndarray:
    """Espectro do perfil de |gradiente| médio por linha/coluna.

    Os harmônicos de uma estrutura periódica vivem aqui, não no PSD da imagem:
    as descontinuidades têm amplitude aproximadamente independente e de média
    zero de bloco para bloco, e um trem de impulsos assim tem PSD **plano**. É a
    não-linearidade do valor absoluto que faz a média de bloco deixar de ser
    zero e a periodicidade aparecer.
    """
    g = np.abs(np.diff(np.asarray(tile, dtype=np.float64), axis=axis)).mean(axis=1 - axis)
    return np.abs(np.fft.rfft((g - g.mean()) * np.hanning(g.size))) ** 2, g.size


def _peak_over_background(spec: np.ndarray, index: int, masked: np.ndarray) -> float:
    bg = float(np.median(spec[1:][~masked[1:]]))
    i = min(int(index), spec.size - 1)
    peak = float(spec[max(0, i - 2):min(i + 3, spec.size)].max())
    return float(np.log10(max(peak, 1e-300) / max(bg, 1e-300)))


def harmonic_excess(tile: np.ndarray, period_px: int) -> tuple[float, dict]:
    """Excesso (decadas) nos harmônicos de 1/`period_px`, **excluindo Nyquist**.

    O harmônico de Nyquist (h = period/2, isto é, período 2 px) é deliberadamente
    excluído: ele dispara com qualquer alternância pixel-a-pixel — xadrez de
    upsampling, reamostragem 2×, ruído de requantização — e não é específico do
    período pedido. Tomá-lo junto com os demais faria uma alternância de 2 px ser
    reportada como "grade de `period_px`", que é precisamente a confusão que este
    projeto precisa evitar. Ele é medido em separado por `nyquist_excess`.

    Para `period_px = 4` sobra apenas h=1 (período 4); para 8, h=1..3 — a mesma
    convenção herdada da bancada anterior.
    """
    size = int(period_px)
    detail: dict[str, float] = {}
    for name, axis in (("x", 1), ("y", 0)):
        spec, m = _gradient_profile_spectrum(tile, axis)
        if m < 4 * size:
            raise ValueError(f"tile pequeno demais ({m} px) para período de {size} px")
        masked = np.zeros(spec.size, dtype=bool)
        for h in range(1, size // 2 + 1):                 # mascara todos, Nyquist incluído
            i = min(int(round(h / size * m)), spec.size - 1)
            masked[max(0, i - 2):i + 3] = True
        for h in range(1, (size + 1) // 2):                # reporta só abaixo de Nyquist
            detail[f"{name}_h{h}"] = _peak_over_background(spec, round(h / size * m), masked)
    if not detail:
        raise ValueError(f"período {size} px não tem harmônico abaixo de Nyquist")
    return max(detail.values()), detail


def nyquist_excess(tile: np.ndarray) -> tuple[float, dict]:
    """Excesso (decadas) na alternância de **período 2 px** do perfil de |gradiente|.

    É a assinatura do xadrez de convolução transposta / pixel-shuffle em
    decodificadores de upsampling 2×. Medido no gen-fid: sintético 0,9–2,1
    decadas, real −0,8 a +0,2 — separa os conjuntos quase perfeitamente, mas é
    **artefato de geração, não textura de tecido**, e por isso é medido e
    reportado sob nome próprio em vez de contaminar uma métrica de textura.
    """
    detail: dict[str, float] = {}
    for name, axis in (("x", 1), ("y", 0)):
        spec, m = _gradient_profile_spectrum(tile, axis)
        masked = np.zeros(spec.size, dtype=bool)
        masked[-3:] = True
        detail[name] = _peak_over_background(spec, spec.size - 1, masked)
    return max(detail.values()), detail


def highfreq_rolloff(radial: dict, fit: dict, band_frac_of_nyquist: Sequence[float]) -> float:
    """Resíduo médio (decadas) entre potência observada e lei de potência extrapolada.

    Negativo e grande = alta frequência sumiu (reamostragem passa-baixa).
    Positivo = piso de ruído / artefato acima da lei de potência.
    """
    lo_f, hi_f = band_frac_of_nyquist
    fnyq = float(radial["f_nyquist_cpx"])
    f, p = radial["freq_cpx"], radial["power"]
    m = (f >= lo_f * fnyq) & (f <= hi_f * fnyq) & (p > 0)
    if int(m.sum()) < 3:
        return float("nan")
    pred = fit["intercept"] - fit["beta"] * np.log10(f[m])
    return float(np.mean(np.log10(p[m]) - pred))


def spectral_scalars(tile: np.ndarray, cfg_psd: dict, fit_bins: Sequence[int] | None = None) -> tuple[dict, dict]:
    """Todos os escalares espectrais de um tile + o perfil radial (para `profiles.parquet`)."""
    psd = power_spectrum_2d(tile, int(cfg_psd["detrend_order"]), cfg_psd["window"] == "hann_periodic")
    rad = radial_average(psd)
    fit = fit_power_law(rad, fit_bins or cfg_psd["fit_bins"])
    out = {"beta": fit["beta"], "r2_fit": fit["r2"], "n_bins_fit": float(fit["n_bins_fit"]),
           "rolloff_decades": highfreq_rolloff(rad, fit, cfg_psd["rolloff_band_frac_of_nyquist"]),
           "total_power": float(rad["power"].sum())}
    out.update(band_fractions(psd, cfg_psd["bands_cpx"]))
    for per in cfg_psd["harmonic_periods_px"]:
        out[f"harmonic_{per}px"] = harmonic_excess(tile, per)[0]
    out["nyquist_excess"] = nyquist_excess(tile)[0]
    return out, rad


def synthesize_power_law_field(n: int, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Campo gaussiano 2D com PSD ∝ f^(−beta): espectro branco × f^(−beta/2), IFFT."""
    f1 = np.fft.fftfreq(n)
    fr = np.hypot(f1[None, :], f1[:, None])
    fr[0, 0] = 1.0
    amp = fr ** (-beta / 2.0)
    amp[0, 0] = 0.0
    z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return np.fft.ifft2(amp * z).real
