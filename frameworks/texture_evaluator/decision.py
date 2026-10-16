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

    if status == "rejeitar":
        # com o brilho fora do aceitável, a textura é registrada mas não é o motivo da decisão
        reasons = [reason for reason in reasons if reason.startswith("brilho")] + [f"(textura não interpretada: {row['wavelet_verdict']})"]
    return {
        "final_status": status,
        "final_reasons": "; ".join(reasons) or "brilho e textura dentro do esperado",
        "glcm_contrast_pct": glcm_pct,
        "texture_agreement": "concordam" if agree else "discordam",
    }
