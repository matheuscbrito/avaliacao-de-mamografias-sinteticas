"""Camada de decisão: junta brilho e textura num veredito final por corte.

Três perguntas, nesta ordem:

1. **Brilho mudou?** Duas leituras lado a lado (a convivência ainda será decidida):
   - filtro do contrato (Δμ, Sat31): é o único que pode **rejeitar**;
   - densidade da Wavelet: só manda para **revisão** (limites ainda exploratórios).
2. **Quanto detalhe sobrou?** Contraste GLCM e energia fina Wavelet, em % da
   original. As duas medem quase a mesma coisa (ρ = 0,99 no Dsr), então uma
   confere a outra: se discordarem, o corte vai para revisão.
3. **Que tipo de defeito?** Veredito Wavelet (razão fino/grosso): borrou,
   ruído, perdeu contraste…
4. **Há assinatura espectral anômala?** NEx medido na mesma ROI pulmonar;
   fora de P5–P95 das originais manda para revisão. Poucas janelas =
   inconclusivo, nunca reprova sozinho.

Resultado: `aprovada`, `revisar` ou `rejeitar`, com os motivos em texto.
Rejeição direta só vem do brilho; defeito de textura manda para revisão,
porque os limites de textura ainda não foram validados.
"""
from __future__ import annotations

# faixa em que a quantidade de detalhe conta como preservada (% da original);
# a mesma do veredito Wavelet (densidade_wavelet.wavelet_verdict) — exploratória
TEXTURE_RANGE = (80.0, 125.0)
DENSITY_TEXT = {"brighter": "clareou", "darker": "escureceu", "too_bright": "claro demais", "too_dark": "escuro demais"}
FINAL_STATUSES = ("aprovada", "revisar", "rejeitar")


# Classificação de cada métrica Wavelet, com a mesma regra do filtro de brilho:
# a diferença sintética − original, medida em IQR daquela métrica entre as CTs
# reais de referência. Até 1 IQR, a mudança cabe na diferença entre dois
# pulmões reais (aprovada); entre 1 e 2, revisão; acima de 2, reprovada.
# Vale só para a leitura de textura: o veredito final continua mandando
# textura para revisão, nunca para rejeição direta.
WAVELET_METRICS = {"fine": "Wavelet 2 px (textura fina)", "coarse": "Wavelet 4 px (textura grossa)", "entropy": "Wavelet entropia"}
# a GLCM segue a mesma regra, com o IQR de cada métrica entre as originais
GLCM_METRICS = {"contrast": "GLCM contraste", "energy": "GLCM energia"}
METRIC_STATUSES = ("aprovada", "revisão", "reprovada")
# NEx mantém a própria regra (sintética fora de P5–P95 das originais = revisão);
# com poucas janelas na ROI a leitura é inconclusiva e não pesa no veredito
NEX_STATUS = {"ok": "aprovada", "alto": "revisão", "baixo": "revisão", "inconclusivo": "inconclusivo"}
TECHNIQUES = ("GLCM", "Wavelet", "NEx")


def metric_status(z: float) -> str:
    return "reprovada" if abs(z) > 2 else "revisão" if abs(z) > 1 else "aprovada"


def worst(statuses) -> str:
    """Pior classificação; "inconclusivo" não conta."""
    return max((s for s in statuses if s in METRIC_STATUSES), key=METRIC_STATUSES.index, default="aprovada")


def metric_columns(prefix: str, metrics: dict, real: dict, synthetic: dict, iqr: dict) -> dict:
    """Valores, diferença em IQR e classificação de cada métrica de uma técnica, mais a pior delas."""
    out = {}
    for key in metrics:
        z = (synthetic[key] - real[key]) / max(iqr[key], 1e-12)
        out.update({f"{prefix}_{key}_real": real[key], f"{prefix}_{key}_synthetic": synthetic[key], f"{prefix}_{key}_z": z, f"{prefix}_{key}_status": metric_status(z)})
    out[f"{prefix}_status"] = worst(out[f"{prefix}_{key}_status"] for key in metrics)
    return out


def wavelet_metric_columns(real: dict, synthetic: dict, iqr: dict) -> dict:
    return metric_columns("wavelet", WAVELET_METRICS, real, synthetic, iqr)


def glcm_metric_columns(real: dict, synthetic: dict, iqr: dict) -> dict:
    return metric_columns("glcm", GLCM_METRICS, real, synthetic, iqr)


def ll_columns(real: float, synthetic: float, iqr: float) -> dict:
    """Brilho segundo a Wavelet (banda LL), na mesma regra de IQR. Entra no veredito de brilho do painel."""
    z = (synthetic - real) / max(iqr, 1e-12)
    return {"wavelet_ll_z": z, "wavelet_ll_status": metric_status(z)}


def technique_statuses(row: dict) -> dict:
    """Classificação de cada técnica de textura neste corte."""
    return {"GLCM": row["glcm_status"], "Wavelet": row["wavelet_status"], "NEx": NEX_STATUS[row.get("nex_status", "inconclusivo")]}


def texture_status(row: dict) -> str:
    """Veredito de textura: a pior das três técnicas. O brilho não entra aqui."""
    return worst(technique_statuses(row).values())


# Veredito de brilho do painel: GLCM (o filtro do contrato, por Δμ e Sat31 na
# régua de 32 tons) e Wavelet (média da banda LL, em IQR das originais). A LL
# acompanha o brilho e pega cortes claros que o Δμ deixa passar; energia e
# entropia da Wavelet não respondem ao brilho e ficam fora daqui.
BRIGHTNESS_TECHNIQUES = ("GLCM", "Wavelet")
GLCM_BRIGHTNESS = {"approved": "aprovada", "manual_review": "revisão", "hard_reject": "reprovada"}


def brightness_statuses(row: dict) -> dict:
    """Classificação de cada técnica de brilho neste corte."""
    return {"GLCM": GLCM_BRIGHTNESS[row["brightness_status"]], "Wavelet": row["wavelet_ll_status"]}


def brightness_verdict(row: dict) -> str:
    """Veredito de brilho: a pior das duas técnicas. A textura não entra aqui."""
    return worst(brightness_statuses(row).values())


def _inside(value: float) -> bool:
    return TEXTURE_RANGE[0] <= value <= TEXTURE_RANGE[1]


def decide(row: dict) -> dict:
    """Recebe uma linha do MVP e devolve as colunas da decisão."""
    reasons, status = [], "aprovada"

    # 1. brilho
    if row["brightness_status"] == "hard_reject":
        status = "rejeitar"
        reasons.append(f"brilho (filtro): rejeição, Δμ={float(row['delta_mu']):+.3f}, Sat31={float(row['sat31_pct']):.1f}%")
    elif row["brightness_status"] == "manual_review":
        status = "revisar"
        reasons.append(f"brilho (filtro): Δμ={float(row['delta_mu']):+.3f}")
    if row["density_status"] != "ok":
        status = max(status, "revisar", key=FINAL_STATUSES.index)
        reasons.append(f"brilho (densidade): {DENSITY_TEXT[row['density_status']]}")

    # 2. quantidade de detalhe: GLCM e Wavelet devem concordar
    glcm_pct = 100 * float(row["glcm_contrast_synthetic"]) / float(row["glcm_contrast_real"])
    fine_pct = float(row["wavelet_fine_pct"])
    agree = _inside(glcm_pct) == _inside(fine_pct)
    if not agree:
        status = max(status, "revisar", key=FINAL_STATUSES.index)
        reasons.append(f"GLCM e Wavelet discordam: contraste {glcm_pct:.0f}%, detalhe fino {fine_pct:.0f}%")

    # 3. tipo de defeito (o veredito Wavelet já inclui a quantidade de detalhe fino e grosso)
    texture_ok = row["wavelet_verdict"] == "textura preservada"
    if not texture_ok:
        status = max(status, "revisar", key=FINAL_STATUSES.index)
        reasons.append(f"textura: {row['wavelet_verdict']} (fino {fine_pct:.0f}%, grosso {float(row['wavelet_coarse_pct']):.0f}%, contraste GLCM {glcm_pct:.0f}%)")

    # NEx é independente das leituras GLCM/Wavelet de energia: ele procura
    # excesso perto de Nyquist nas janelas internas da ROI. É um gatilho de
    # revisão, não de rejeição direta, enquanto a validação clínica é limitada.
    nex_status = row.get("nex_status", "inconclusivo")
    if nex_status in {"alto", "baixo"}:
        status = max(status, "revisar", key=FINAL_STATUSES.index)
        reasons.append(f"NEx: {nex_status} fora da faixa real (Δ {float(row['nex_delta']):+.3f})")
    elif nex_status == "inconclusivo":
        reasons.append("NEx: inconclusivo (poucas janelas na ROI)")

    if status == "rejeitar":
        # com o brilho fora do aceitável, a textura é registrada mas não é o motivo da decisão
        reasons = [reason for reason in reasons if reason.startswith("brilho")] + [f"(textura não interpretada: {row['wavelet_verdict']})"]
    return {
        "final_status": status,
        "final_reasons": "; ".join(reasons) or "brilho e textura dentro do esperado",
        "glcm_contrast_pct": glcm_pct,
        "texture_agreement": "concordam" if agree else "discordam",
    }
