# Placar por versão

37 cortes avaliados em todas as versões. Réguas calibradas nas originais de `Dgen`.
Valor = mediana entre os cortes (ou % dos cortes); entre colchetes, intervalo de 95% (bootstrap).

## Versões

| Versão | Pasta | Modo | Pares suspeitos | Adicionada em | Cortes válidos |
|---|---|---|---|---|---|
| Dgen | dados/Dgen | só máscara | 0 | 2026-10-15 22:08:27 | 37 |
| Dsr | dados/Dsr | mesma anatomia | 0 | 2026-10-15 22:08:29 | 37 |

## Veredito final

| Indicador | Ideal | Dgen | Dsr |
|---|---|---|---|
| Aprovadas (% dos cortes) | 100% | 3% [0%, 8%] | 14% [3%, 27%] |

| Mudança | Dgen → Dsr |
|---|---|
| Aprovadas | melhorou (+10.8 pp) |

| Contagem de cortes | Dgen | Dsr |
|---|---|---|
| aprovada | 1 | 5 |
| rejeitar: brilho | 5 | 0 |
| revisar: borrou | 9 | 27 |
| revisar: perdeu contraste | 13 | 5 |
| revisar: estourou | 3 | 0 |
| revisar: ruído | 2 | 0 |
| revisar: só brilho | 2 | 0 |
| revisar: ganhou contraste | 2 | 0 |

## Brilho

| Indicador | Ideal | Dgen | Dsr |
|---|---|---|---|
| \|Δμ\| (desvio de brilho) (\|sintética − original\|) | 0.000 | 0.310 [0.211, 0.432] | 0.143 [0.065, 0.215] |
| Brilho sinalizado (% dos cortes) | 0% | 46% [30%, 62%] | 8% [0%, 19%] |
| Área com brilho local alterado (% do pulmão) | 0% | n/a | 0% [0%, 5%] |

| Mudança | Dgen → Dsr |
|---|---|
| \|Δμ\| (desvio de brilho) | melhorou (+0.095) |
| Brilho sinalizado | melhorou (+37.8 pp) |
| Área com brilho local alterado | n/a |

| Contagem de cortes | Dgen | Dsr |
|---|---|---|
| densidade: ok | 20 | 34 |
| densidade: clareou | 9 | 0 |
| densidade: escureceu | 5 | 2 |
| densidade: claro demais | 3 | 0 |
| densidade: escuro demais | 0 | 1 |
| filtro: aprovado | 21 | 35 |
| filtro: revisão | 11 | 2 |
| filtro: rejeição | 5 | 0 |

## Textura

| Indicador | Ideal | Dgen | Dsr |
|---|---|---|---|
| Detalhe fino (Wavelet) (% da original) | 100% | 58% [44%, 84%] | 54% [49%, 62%] |
| Detalhe grosso (Wavelet) (% da original) | 100% | 60% [44%, 72%] | 66% [60%, 70%] |
| Razão fino/grosso (1 = sem borrar nem ruído) | 1.00 | 0.98 [0.91, 1.04] | 0.83 [0.82, 0.86] |
| Contraste GLCM (% da original) | 100% | 61% [50%, 89%] | 60% [54%, 65%] |
| Textura alterada (% dos cortes) | 0% | 86% [76%, 97%] | 86% [76%, 97%] |
| Área com textura alterada (% do pulmão) | 0% | n/a | 93% [87%, 94%] |
| Correlação espacial da textura (1 = textura nos mesmos lugares) | 1.00 | n/a | 0.87 [0.83, 0.89] |

| Mudança | Dgen → Dsr |
|---|---|
| Detalhe fino (Wavelet) | sem diferença clara (+5.1 pp) |
| Detalhe grosso (Wavelet) | melhorou (+11.3 pp) |
| Razão fino/grosso | piorou (-0.09) |
| Contraste GLCM | sem diferença clara (+4.2 pp) |
| Textura alterada | sem diferença clara (+0.0 pp) |
| Área com textura alterada | n/a |
| Correlação espacial da textura | n/a |

| Contagem de cortes | Dgen | Dsr |
|---|---|---|
| borrou | 10 | 27 |
| estourou | 3 | 0 |
| ganhou contraste | 2 | 0 |
| perdeu contraste | 14 | 5 |
| ruído | 3 | 0 |
| textura preservada | 5 | 5 |

Ganho = quanto, corte a corte, a versão nova chegou mais perto do ideal (mediana; em taxas, pontos percentuais). "melhorou" ou "piorou" só quando o intervalo de 95% não cruza o zero e a mudança passa do mínimo relevante (5 pp nas porcentagens e taxas; 0,05 na razão, na correlação e no |Δμ|); fora do zero mas abaixo do mínimo: "mudança pequena". n/a: indicador do mapa em versão no modo só máscara.
