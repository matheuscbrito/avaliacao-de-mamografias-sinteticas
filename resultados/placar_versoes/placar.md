# Placar por versão

37 cortes avaliados em todas as versões. Réguas calibradas nas originais de `Dgen`.
Valor = mediana entre os cortes; entre colchetes, intervalo de 95% (bootstrap).

## Indicadores

| Indicador | Ideal | Dgen | Dsr |
|---|---|---|---|
| Detalhe fino (Wavelet) (% da original) | 100% | 58% [44%, 84%] | 54% [49%, 62%] |
| Detalhe grosso (Wavelet) (% da original) | 100% | 60% [44%, 69%] | 66% [60%, 70%] |
| Razão fino/grosso (1 = sem borrar nem ruído) | 1.00 | 0.98 [0.91, 1.04] | 0.83 [0.82, 0.86] |
| Contraste GLCM (% da original) | 100% | 61% [50%, 89%] | 60% [54%, 65%] |
| Δμ (brilho) (sintética − original) | +0.000 | -0.093 [-0.211, +0.170] | -0.143 [-0.215, -0.046] |
| Aprovadas (% dos cortes) | 100% | 3% [0, 8] | 14% [3, 24] |
| Brilho sinalizado (% dos cortes) | 0% | 46% [30, 62] | 8% [0, 19] |

## Vereditos

| Veredito | Dgen | Dsr |
|---|---|---|
| revisar: borrou | 9 | 27 |
| revisar: perdeu contraste | 13 | 5 |
| aprovada | 1 | 5 |
| rejeitar: brilho | 5 | 0 |
| revisar: estourou | 3 | 0 |
| revisar: ruído | 2 | 0 |
| revisar: ganhou contraste | 2 | 0 |
| revisar: só brilho | 2 | 0 |

## Comparação com a versão anterior

Ganho = quanto, corte a corte, a versão nova chegou mais perto do ideal (mediana; em taxas, pontos percentuais).
Só é "melhorou" ou "piorou" quando o intervalo de 95% não cruza o zero **e** a mudança passa do mínimo relevante
(5 pp nas porcentagens e taxas, 0,05 na razão e no Δμ); se só a primeira condição vale, aparece "mudança pequena".

| Indicador | Dgen → Dsr |
|---|---|
| Detalhe fino (Wavelet) | **sem diferença clara** (+5.1 pp) |
| Detalhe grosso (Wavelet) | **melhorou** (+11.3 pp) |
| Razão fino/grosso | **piorou** (-0.09) |
| Contraste GLCM | **sem diferença clara** (+4.2 pp) |
| Δμ (brilho) | **melhorou** (+0.09) |
| Aprovadas | **melhorou** (+10.8 pp) |
| Brilho sinalizado | **melhorou** (+37.8 pp) |
