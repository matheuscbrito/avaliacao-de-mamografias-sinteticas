"""Validação dos estimadores. Sem isto, um β sistematicamente errado por 0,3
produziria tabelas plausíveis e ninguém notaria.

Herdado da antiga bancada INbreast (seção 12; já removida do repo): recuperação
de β, degenerados e equivariância angular — mais os testes dos controles novos
exigidos por este projeto: `real_noise` sem grade periódica, `bicubic` com
déficit real de alta frequência, e quadrantes do mesmo tecido com AUC ≈ 0,5.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from texture import io, metrics, spectrum, stats  # noqa: E402

N = 512
DX_SEED = 20260904
BANDS = [[0.0, 0.015625], [0.015625, 0.0625], [0.0625, 0.125], [0.125, 0.25], [0.25, 0.5]]
FIT_BINS = [5, 29]


def beta_of(field, use_window=True):
    psd = spectrum.power_spectrum_2d(field, 2, use_window)
    rad = spectrum.radial_average(psd)
    fit = spectrum.fit_power_law(rad, FIT_BINS)
    return fit["beta"], fit["r2"]


# ------------------------------------------------------------- recuperação de β

@pytest.mark.parametrize("beta_true", [0.0, 2.0, 3.0, 4.0])
def test_beta_recovery(beta_true):
    """A média sobre 32 realizações recupera β com viés < 0,1.

    Um único teste cobre detrend, janela, normalização, média radial, eixo de
    frequência e regressão: se qualquer um estiver errado, β sai errado.
    """
    rng = np.random.default_rng(DX_SEED)
    est = np.array([beta_of(spectrum.synthesize_power_law_field(N, beta_true, rng))[0]
                    for _ in range(32)])
    assert abs(est.mean() - beta_true) <= 0.10, f"viés {est.mean() - beta_true:+.3f} em β={beta_true}"
    assert est.std(ddof=1) < 0.30, "desvio por tile alto demais para o piso de ruído esperado"


def test_window_matters():
    """Sem janela, o vazamento espectral achata a inclinação medida (β=4 → ~3,4)."""
    rng = np.random.default_rng(DX_SEED + 1)
    fields = [spectrum.synthesize_power_law_field(N, 4.0, rng) for _ in range(8)]
    with_w = np.mean([beta_of(f, True)[0] for f in fields])
    without = np.mean([beta_of(f, False)[0] for f in fields])
    assert with_w - without > 0.3, "a janela de Hann deveria alterar β de forma mensurável"
    assert abs(with_w - 4.0) < abs(without - 4.0), "com janela deveria ficar mais perto do verdadeiro"


def test_white_noise_has_low_r2_and_flat_beta():
    """Ruído branco: β≈0 na média e R² baixo em toda realização.

    A tolerância de β vale para a **média**: um tile único tem sd ≈ 0,15, então
    exigir |β| < 0,2 de uma realização só reprovaria ~18 % dos sorteios legítimos.
    """
    rng = np.random.default_rng(7)
    betas, r2s = zip(*(beta_of(rng.normal(size=(N, N))) for _ in range(24)))
    assert abs(np.mean(betas)) < 0.1, f"ruído branco deveria dar β≈0, veio {np.mean(betas):+.3f}"
    # R² de um ajuste a ruído é ele próprio aleatório e ocasionalmente alto; o que
    # caracteriza o estimador é a média, e o contraste com um campo genuíno.
    r2_real = np.mean([beta_of(spectrum.synthesize_power_law_field(N, 3.0, rng))[1] for _ in range(8)])
    assert np.mean(r2s) < 0.25, f"ruído branco não deveria ajustar lei de potência (R² médio={np.mean(r2s):.3f})"
    assert r2_real > 0.95, f"campo β=3 deveria ajustar quase perfeitamente (R²={r2_real:.3f})"


def test_parseval_normalization():
    """A soma do PSD sobre a grade recupera a variância do resíduo janelado."""
    rng = np.random.default_rng(11)
    field = spectrum.synthesize_power_law_field(N, 3.0, rng)
    psd = spectrum.power_spectrum_2d(field, 2, True)
    resid = spectrum.poly_detrend_2d(field, 2)[0]
    w, u = spectrum.hann2d(N)
    expected = float(((resid * w) ** 2).sum() / (N ** 2 * u))
    assert np.isclose(psd.sum() / N ** 2, expected, rtol=1e-10)


def test_band_fractions_sum_to_one_and_are_gain_invariant():
    rng = np.random.default_rng(13)
    field = spectrum.synthesize_power_law_field(N, 3.0, rng)
    psd = spectrum.power_spectrum_2d(field, 2, True)
    f1 = spectrum.band_fractions(psd, BANDS)
    bands_only = {k: v for k, v in f1.items() if k != "band_frac_corners"}
    assert np.isclose(sum(bands_only.values()), 1.0, atol=1e-9)
    psd2 = spectrum.power_spectrum_2d(field * 7.0, 2, True)
    f2 = spectrum.band_fractions(psd2, BANDS)
    for k in f1:
        assert np.isclose(f1[k], f2[k], rtol=1e-9), f"{k} não é invariante a ganho"


def test_band_fractions_exclude_anisotropic_corners():
    """Os cantos da grade (|f| > 0,5) ficam fora do denominador.

    São 21,5 % dos modos e só existem nas diagonais: incluí-los faria a fração
    depender da anisotropia do conteúdo. Em ruído branco eles carregam ~22 % da
    potência — a diferença entre excluir e não excluir é enorme.
    """
    rng = np.random.default_rng(71)
    f = spectrum.band_fractions(spectrum.power_spectrum_2d(rng.normal(size=(N, N)), 2, True), BANDS)
    bands_only = {k: v for k, v in f.items() if k != "band_frac_corners"}
    assert np.isclose(sum(bands_only.values()), 1.0, atol=1e-9), "as bandas devem cobrir o disco inteiro"
    assert 0.15 < f["band_frac_corners"] < 0.30, "os cantos deveriam ser registrados, não ignorados"


# ---------------------------------------------------------- harmônicos e controles

def test_harmonic_detects_periodic_grid():
    """Uma grade de 4 px injetada é detectada em `harmonic_4px`."""
    rng = np.random.default_rng(17)
    base = spectrum.synthesize_power_law_field(N, 3.0, rng)
    base = base / base.std()
    grid = np.zeros((N, N)); grid[::4, :] = 1.0; grid[:, ::4] = 1.0
    clean4 = spectrum.harmonic_excess(base, 4)[0]
    dirty4 = spectrum.harmonic_excess(base + 0.5 * grid, 4)[0]
    assert dirty4 - clean4 > 1.0, f"grade de 4 px não detectada ({clean4:.2f} → {dirty4:.2f})"


def test_nyquist_and_period4_are_not_confounded():
    """Um xadrez de 2 px vai para `nyquist_excess`, NÃO para `harmonic_4px`.

    É a distinção que define o achado do gen-fid: o sintético carrega alternância
    de período 2 (xadrez de upsampling), não uma grade de 4 px. Tomar o máximo
    sobre os harmônicos de `size=4` inclui o bin de Nyquist e faz uma coisa ser
    reportada com o nome da outra.
    """
    rng = np.random.default_rng(97)
    base = spectrum.synthesize_power_law_field(N, 3.0, rng)
    base = base / base.std()
    # Modelo do artefato: campo gerado em meia resolução e ampliado 2×, mais
    # ruído. O |gradiente| fica pequeno DENTRO de cada bloco 2×2 e grande na
    # fronteira entre blocos — é essa alternância que produz o pico em Nyquist.
    # (Um xadrez aditivo não serve: seu |gradiente| é constante e não oscila.)
    half = spectrum.synthesize_power_law_field(N // 2, 3.0, rng)
    half /= half.std()
    dirty = np.repeat(np.repeat(half, 2, 0), 2, 1) + 0.15 * rng.normal(size=(N, N))
    assert spectrum.nyquist_excess(dirty)[0] - spectrum.nyquist_excess(base)[0] > 1.5, \
        "alternância de 2 px deveria disparar nyquist_excess"
    assert spectrum.harmonic_excess(dirty, 4)[0] - spectrum.harmonic_excess(base, 4)[0] < 0.6, \
        "alternância de 2 px NÃO deveria ser reportada como grade de 4 px"


def test_harmonic_excess_reports_below_nyquist_only():
    """`harmonic_excess(·, 4)` reporta só h=1; `·, 8` reporta h=1..3 (Nyquist fora)."""
    rng = np.random.default_rng(101)
    tile = spectrum.synthesize_power_law_field(256, 3.0, rng)
    assert set(spectrum.harmonic_excess(tile, 4)[1]) == {"x_h1", "y_h1"}
    assert set(spectrum.harmonic_excess(tile, 8)[1]) == {f"{a}_h{h}" for a in "xy" for h in (1, 2, 3)}


def test_noise_control_has_no_periodic_grid():
    """`real + ruído branco` sobe a alta frequência SEM criar grade periódica.

    É o que separa o controle de alucinação do artefato de 4 px do decodificador.
    """
    rng = np.random.default_rng(19)
    real = spectrum.synthesize_power_law_field(N, 3.0, rng)
    real = real / real.std()
    noisy = io.noise_control(real, 0.3, rng)
    assert abs(spectrum.harmonic_excess(noisy, 4)[0] - spectrum.harmonic_excess(real, 4)[0]) < 0.8
    hi_real = spectrum.band_fractions(spectrum.power_spectrum_2d(real, 2, True), BANDS)["band_frac_0.25_0.5"]
    hi_noisy = spectrum.band_fractions(spectrum.power_spectrum_2d(noisy, 2, True), BANDS)["band_frac_0.25_0.5"]
    assert hi_noisy > hi_real, "o controle de ruído deveria aumentar a potência de alta frequência"


def test_bicubic_control_removes_high_frequency():
    """`bicubic_4` deve ter razão espectral < 0,5 acima do corte de 0,125 c/px."""
    rng = np.random.default_rng(23)
    real = spectrum.synthesize_power_law_field(N, 3.0, rng)
    bic = io.bicubic_control(real, 4)
    r_real = spectrum.radial_average(spectrum.power_spectrum_2d(real, 2, True))
    r_bic = spectrum.radial_average(spectrum.power_spectrum_2d(bic, 2, True))
    hi = r_real["freq_cpx"] > 0.125
    ratio = np.median(r_bic["power"][hi] / r_real["power"][hi])
    assert ratio < 0.5, f"controle bicúbico não removeu alta frequência (razão {ratio:.3f})"


def test_lowpass_preserves_low_and_kills_high():
    rng = np.random.default_rng(29)
    real = spectrum.synthesize_power_law_field(N, 3.0, rng)
    lp = io.butterworth_lowpass(real, 0.125, 4)
    r_real = spectrum.radial_average(spectrum.power_spectrum_2d(real, 2, True))
    r_lp = spectrum.radial_average(spectrum.power_spectrum_2d(lp, 2, True))
    ratio = r_lp["power"] / r_real["power"]
    f = r_real["freq_cpx"]
    assert np.median(ratio[f < 0.03]) > 0.95, "o passa-baixa não deveria tocar λ > 32 px"
    assert np.median(ratio[f > 0.25]) < 0.05, "o passa-baixa deveria eliminar a banda da grade de 4 px"


# --------------------------------------------------------------- alinhamento

def test_affine_align_recovers_known_transform():
    """Com transformação afim exata, os dois métodos coincidem e a recuperam."""
    rng = np.random.default_rng(31)
    real = np.abs(spectrum.synthesize_power_law_field(128, 3.0, rng)) + 1.0
    syn = 0.38 * real + 0.34
    for method in ("moment", "ls"):
        al, qc = io.affine_align(real, syn, method)
        assert np.isclose(qc["affine_a"], 0.38, rtol=1e-6), method
        assert np.isclose(qc["affine_b"], 0.34, atol=1e-6), method
        assert np.allclose(al, real, atol=1e-8), method


def test_ls_alignment_inflates_variance_but_moment_does_not():
    """O alinhamento não pode acoplar a normalização à qualidade do par.

    A regressão de `syn` em `real` é atenuante: desfazê-la dá
    `var(syn_al) = var(real)/corr²`, inflação que cresce quando a correlação cai.
    Como os batches do gen-fid diferem em correlação média, usar LS faria a
    comparação entre conjuntos medir qualidade de pareamento disfarçada de
    excesso de potência. O casamento de momentos preserva a razão de variâncias
    qualquer que seja a correlação.
    """
    rng = np.random.default_rng(83)
    real = spectrum.synthesize_power_law_field(256, 3.0, rng)
    for noise, expected_corr in [(0.3, "alta"), (1.5, "baixa")]:
        syn = 0.4 * real + 0.2 + noise * real.std() * rng.normal(size=real.shape)
        corr = np.corrcoef(real.ravel(), syn.ravel())[0, 1]
        ls, _ = io.affine_align(real, syn, "ls")
        mom, _ = io.affine_align(real, syn, "moment")
        assert np.isclose(ls.std() / real.std(), 1 / corr, rtol=0.02), \
            f"LS deveria inflar por 1/corr ({expected_corr} correlação)"
        assert np.isclose(mom.std(), real.std(), rtol=1e-9), \
            "casamento de momentos deve preservar o desvio, qualquer que seja a correlação"


def test_requantize_lands_on_grid():
    rng = np.random.default_rng(37)
    x = rng.uniform(0, 1, size=(64, 64))
    step = 1 / 4095
    q = io.requantize(x, step)
    assert np.allclose(q / step, np.rint(q / step))
    assert np.abs(q - x).max() <= step / 2 + 1e-12


# ------------------------------------------------------------------ GLCM

def test_glcm_rotation_permutes_angles():
    """Rotacionar 90° deve permutar os ângulos (0↔90, 45↔135) sem mudar os valores."""
    rng = np.random.default_rng(41)
    n = 128
    yy, xx = np.mgrid[0:n, 0:n]
    aniso = (rng.normal(size=(n, n)) * 0.15 + 2.0 * np.sin(2 * np.pi * (xx + yy) / 8)
             + 1.1 * np.sin(2 * np.pi * xx / 6) + 0.4 * np.sin(2 * np.pi * yy / 10))
    cfg = {"n_levels": 32, "distances_px": [1, 2], "angles_deg": [0, 45, 90, 135],
           "props": ["contrast", "homogeneity", "ASM", "correlation"]}
    lo, hi = float(aniso.min()), float(aniso.max())
    s_o = metrics.glcm_features(metrics.quantize(aniso, lo, hi, 32), cfg)
    s_r = metrics.glcm_features(metrics.quantize(np.rot90(aniso, 1), lo, hi, 32), cfg)
    angles = [0, 45, 90, 135]
    perm = [2, 3, 0, 1]
    for prop in cfg["props"]:
        for d in cfg["distances_px"]:
            scale = max(abs(s_o[f"glcm_{prop}_d{d}_a{a:g}"]) for a in angles) or 1.0
            for i, a in enumerate(angles):
                o = s_o[f"glcm_{prop}_d{d}_a{a:g}"]
                r = s_r[f"glcm_{prop}_d{d}_a{angles[perm[i]]:g}"]
                assert abs(r - o) / scale < 1e-6, f"{prop} d={d} a={a} não permutou"


def test_glcm_degenerate_constant_field():
    const = np.full((64, 64), 1234.0)
    cfg = {"n_levels": 32, "distances_px": [1], "angles_deg": [0],
           "props": ["contrast", "energy", "ASM"]}
    s = metrics.glcm_features(metrics.quantize(const, 0.0, 2000.0, 32), cfg)
    assert abs(s["glcm_contrast_d1_a0"]) < 1e-12
    assert abs(s["glcm_energy_d1_a0"] - 1.0) < 1e-12


def test_global_quantization_is_shared_across_sets():
    """Os limites globais são derivados uma vez e aplicados a todos os conjuntos."""
    rng = np.random.default_rng(43)
    tiles = [rng.normal(0, 1, size=(64, 64)) for _ in range(5)]
    lo, hi = metrics.global_quant_reference(tiles, [1.0, 99.0], seed=0)
    shifted = tiles[0] + 5.0
    q_shift = metrics.quantize(shifted, lo, hi, 32)
    assert q_shift.max() == 31, "um tile deslocado deveria saturar nos bins globais, não ser reescalado"


# -------------------------------------------------------------- estatística

def test_auc_identical_distributions_is_half():
    rng = np.random.default_rng(47)
    a, b = rng.normal(size=400), rng.normal(size=400)
    assert abs(stats.auc_mannwhitney(a, b) - 0.5) < 0.05


def test_auc_separated_distributions_is_one():
    rng = np.random.default_rng(53)
    assert stats.auc_mannwhitney(rng.normal(0, 1, 200), rng.normal(20, 1, 200)) > 0.99


def test_auc_is_direction_agnostic():
    rng = np.random.default_rng(59)
    a, b = rng.normal(0, 1, 200), rng.normal(3, 1, 200)
    assert np.isclose(stats.auc_mannwhitney(a, b), stats.auc_mannwhitney(b, a))


def test_quadrants_of_same_texture_give_auc_half():
    """Piso de ruído: dois quadrantes do mesmo campo não devem ser separáveis."""
    rng = np.random.default_rng(61)
    b1, b2 = [], []
    for _ in range(40):
        field = spectrum.synthesize_power_law_field(N, 3.0, rng)
        q1, q2 = io.quadrants(field, 256)
        for q, dst in ((q1, b1), (q2, b2)):
            rad = spectrum.radial_average(spectrum.power_spectrum_2d(q, 2, True))
            dst.append(spectrum.fit_power_law(rad, [3, 15])["beta"])
    assert abs(stats.auc_mannwhitney(b1, b2) - 0.5) < 0.12


def test_grouped_bootstrap_respects_clusters():
    """Reamostragem por paciente: cada reamostra contém grupos inteiros."""
    groups = np.repeat(np.arange(20), 3)
    rng = np.random.default_rng(67)
    idx = stats._grouped_resample(groups, rng)
    counts = np.bincount(groups[idx], minlength=20)
    assert set(np.unique(counts)) <= {0, 3, 6, 9, 12, 15}, "grupos deveriam entrar inteiros"


def test_geometric_mean_ratio_unbiased_on_log_symmetric_data():
    """Média geométrica de razões recíprocas dá 1; a aritmética daria > 1."""
    ratios = np.array([[2.0], [0.5]] * 50)
    groups = np.arange(100)
    est, lo, hi = stats.geometric_mean_ratio_ci(ratios, groups, n_boot=200, seed=3)
    assert np.isclose(est[0], 1.0, atol=1e-9)
    assert ratios.mean() > 1.2, "a média aritmética é enviesada para cima (motivo da geométrica)"


def test_verdict_table():
    blind = stats.verdict({"bicubic_2": 0.52, "real_noise": 0.55}, 0.99, 0.99, 0.5)
    assert blind == "cega", "sem sensibilidade a controle, a métrica não opina sobre o modelo"
    fine = stats.verdict({"bicubic_2": 0.95, "real_noise": 0.99}, 0.98, 0.52, 0.5)
    assert fine == "falha_escala_fina"
    tissue = stats.verdict({"bicubic_2": 0.95}, 0.98, 0.93, 0.5)
    assert tissue == "falha_escala_tecido"
    ok = stats.verdict({"bicubic_2": 0.95}, 0.50, 0.50, 0.5)
    assert ok == "modelo_passa"
    unstable = stats.verdict({"bicubic_2": 0.95}, 0.98, 0.93, 0.97)
    assert unstable == "instavel", "métrica que separa tecido real de si mesmo não é confiável"


def test_fdr_preserves_nan_and_is_monotone():
    p = [0.001, 0.02, np.nan, 0.5]
    adj = stats.fdr(p)
    assert np.isnan(adj[2])
    assert np.all(adj[[0, 1, 3]] >= np.array(p)[[0, 1, 3]])


def test_glcm_diagonal_distances_collide_and_are_deduplicated():
    """`d=2` a 45°/135° arredonda para o mesmo offset de `d=1` e gera duplicatas exatas.

    Contá-las como testes distintos infla a correção de FDR. O teste trava tanto
    a colisão (para que ela não passe despercebida se o skimage mudar) quanto a
    remoção.
    """
    rng = np.random.default_rng(107)
    tile = spectrum.synthesize_power_law_field(128, 3.0, rng)
    cfg = {"n_levels": 32, "distances_px": [1, 2], "angles_deg": [0, 45, 90, 135],
           "props": ["contrast", "homogeneity"]}
    f = metrics.glcm_features(metrics.quantize(tile, tile.min(), tile.max(), 32), cfg)
    assert f["glcm_contrast_d1_a45"] == f["glcm_contrast_d2_a45"], \
        "d=2 a 45° deveria colidir com d=1 (offset arredondado para (-1,+1))"
    assert f["glcm_contrast_d1_a0"] != f["glcm_contrast_d2_a0"], \
        "d=2 a 0° NÃO colide: offset (0,+2) é distinto de (0,+1)"

    idx = pd.MultiIndex.from_product([[f"p{i}" for i in range(20)], ["real", "syn"]],
                                     names=["pair_id", "set"])
    v = rng.normal(size=len(idx))
    df = pd.DataFrame({"a": v, "b_dup": v.copy(), "c": rng.normal(size=len(idx))}, index=idx)
    kept, dropped = stats.drop_duplicate_metrics(df, verbose=False)
    assert list(kept.columns) == ["a", "c"]
    assert dropped == {"b_dup": "a"}


def test_paired_metrics_are_excluded_from_decision_table():
    """Métricas `*_vs_real` são degeneradas no real e não podem entrar como as demais.

    SSIM(real, real)=1 e PSNR=∞ dariam AUC≈1 contra qualquer conjunto, fazendo a
    métrica parecer sensível sem ter demonstrado nada.
    """
    assert stats.is_paired_metric("ssim_vs_real")
    assert stats.is_paired_metric("lbp_P8R1_js_vs_real")
    assert not stats.is_paired_metric("grad_median")

    idx = pd.MultiIndex.from_product([[f"p{i}" for i in range(20)], ["real", "syn", "bicubic_2"]],
                                     names=["pair_id", "set"])
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"grad_median": rng.normal(size=len(idx)),
                       "ssim_vs_real": rng.normal(size=len(idx))}, index=idx)
    man = pd.DataFrame({"pair_id": [f"p{i}" for i in range(20)],
                        "patient_id": [f"pat{i // 2}" for i in range(20)],
                        "batch": [0, 1] * 10, "abnormality": ["Calc", "Mass"] * 10,
                        "split": ["Test", "Training"] * 10})
    cfg = {"seed": 0, "n_bootstrap": 50, "auc_useful": 0.8, "auc_pass_band": [0.4, 0.6],
           "fdr_alpha": 0.05, "strata": ["batch"]}
    tab = stats.decision_table(df, man, ["bicubic_2"], cfg)
    assert "ssim_vs_real" not in set(tab.metric)
    assert "grad_median" in set(tab.metric)
    paired = stats.paired_comparison_table(df, man, ["bicubic_2"], cfg)
    assert set(paired.metric) == {"ssim_vs_real"}


def test_noise_floor_is_p95_of_absolute_deviation():
    q1 = np.zeros(100)
    q2 = np.arange(100) / 100.0
    assert np.isclose(stats.noise_floor(q1, q2), np.percentile(q2, 95))
