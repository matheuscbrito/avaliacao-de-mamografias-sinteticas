# Contrato do filtro de brilho excessivo

O filtro de brilho é um componente separado do avaliador de textura. Na futura orquestração, ele deverá rodar antes das métricas complexas; esta pasta não implementa essa integração.

Entradas: imagem real, imagem sintética, ROI pulmonar erodida e régua de 32 tons calibrada nos reais. Saídas: `delta_mu`, `sat31_pct` e decisão (`approved`, `manual_review` ou `hard_reject`).

Regras estabelecidas para a coorte atual:

- aprovada: `|Δμ| ≤ 0,314`;
- revisão manual: `0,314 < |Δμ| ≤ 0,629`;
- rejeição direta: `|Δμ| > 0,629` ou `Sat31% > 15%`.

GLCM não deve ser usado como decisão primária desse filtro: contraste e energia não informam a direção da mudança de brilho.
