# Régua única: a textura fica cega ao brilho?

**Data:** 15/10/2026
**Dataset:** padrão (`dados/Dsr`), 37 pares válidos.
**Script:** `frameworks/texture_evaluator/teste_regua.py`

A GLCM e a Wavelet cortavam os tons com réguas diferentes:

| Régua | Faixa | Usada por |
|---|---|---|
| P1–P99 dos pixels reais | −0,44 a 1,58 | Wavelet (antes) |
| Tukey k=3 nos pixels reais | −1,33 a 1,39 | GLCM |

O teste mede as duas métricas com cada régua.

## 1. Brilho puro (simulado)

Figura: `figuras/efeito_brilho_por_regua.png`.

Soma brilho nas originais (±0,5 e ±1 IQR) e mede quanto a textura muda, em IQR entre originais. Brilho puro não muda a textura, então o certo é ~0.

- **P1–P99:** ao escurecer 1 IQR, a energia GLCM sobe **+6,2 IQR** e a entropia wavelet cai **−1,8 IQR**. O piso da régua (−0,44) é alto: o pulmão escuro "cai no chão" e vira um platô. Isso é artefato da régua, não textura.
- **Tukey k=3:** todo efeito de brilho fica abaixo de 0,4 IQR. As métricas ficam cegas ao brilho, como deveriam.
- **Controles** (contraste ×0,5 e suavização σ=1): reagem igual nas duas réguas. A régua nova não perde sensibilidade ao que importa. A razão fino/grosso continua sendo a única que separa suavização (−2,1) de contraste (−0,2).

## 2. Dados reais

| | P1–P99 | Tukey k=3 |
|---|---|---|
| Pixels sintéticos no piso (mediana / máximo) | 0,8% / **35%** | 0% / 0% |
| Pixels no teto (mediana, original) | 0,5% | 1,0% |
| Correlação energia GLCM × Δμ | −0,86 | −0,74 |
| Correlação entropia wavelet × Δμ | 0,82 | 0,76 |

- Com P1–P99, uma sintética chegou a ter 35% do pulmão cortado no piso. Com Tukey, nenhuma.
- **A correlação com o Δμ continua mesmo com Tukey, sem nenhum corte no piso.** Logo, ela não vem da régua. Nos dados reais, as sintéticas que escurecem também são as que mudaram de textura: o mesmo defeito do gerador mexe nas duas coisas. A parte 1 mostra que as métricas, sozinhas, não confundem brilho com textura.

## Decisão aplicada

No MVP (`run_mvp.py`), a Wavelet passou a usar a **mesma régua Tukey k=3 da GLCM**. A densidade continua em P1–P99, porque ali a faixa é a própria definição de "fora da faixa real".

Nos 37 pares do Dsr, **nenhum veredito da Wavelet mudou** (27 borrou, 5 perdeu contraste, 5 textura preservada).

## Cuidados

- Clarear mais que +1 IQR começa a saturar o teto da régua Tukey (vasos). O efeito ainda é pequeno (−0,4 IQR na energia grossa).
- Os limites do veredito Wavelet (80–125%, razão 0,9–1,1) continuam exploratórios.
- O código da pasta `wavelet/` (análises anteriores) continua com P1–P99. Só o MVP mudou.
