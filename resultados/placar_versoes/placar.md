# Placar por versão

37 cortes avaliados em todas as versões. Réguas calibradas nas originais de `Dgen`.
Valor = mediana entre os cortes (ou % dos cortes); entre colchetes, intervalo de 95% (bootstrap).

## Versões

| Versão | Pasta | Modo | Pares suspeitos | Adicionada em | Cortes válidos |
|---|---|---|---|---|---|
| Dgen | dados/Dgen | só máscara | 0 | 2026-10-15 22:08:27 | 37 |
| Dsr | dados/Dsr | mesma anatomia | 0 | 2026-10-15 22:08:29 | 37 |

## Veredito final

| Técnica | O que mede | Unidade | Ideal | Dgen | Dsr | Dgen → Dsr |
|---|---|---|---|---|---|---|
| Veredito final | cortes aprovados (sem sinal de brilho nem de textura) | % dos cortes | 100% | 3% [0%, 8%] | 14% [3%, 24%] | melhorou (+10.8 pp) |

| Contagem de cortes | Dgen | Dsr |
|---|---|---|
| aprovada | 1 | 5 |
| rejeitar: brilho | 5 | 0 |
| revisar: borrou | 9 | 27 |
| revisar: perdeu contraste | 13 | 5 |
| revisar: estourou | 3 | 0 |
| revisar: só brilho | 2 | 0 |
| revisar: ganhou contraste | 2 | 0 |
| revisar: ruído | 2 | 0 |

## Brilho

| Técnica | O que mede | Unidade | Ideal | Dgen | Dsr | Dgen → Dsr |
|---|---|---|---|---|---|---|
| Filtro de brilho: Δμ | diferença da média de todos os pixels do pulmão | \|sintética − original\| | 0.000 | 0.310 [0.211, 0.432] | 0.143 [0.065, 0.215] | melhorou (+0.095) |
| Filtro de brilho + densidade | cortes com brilho fora do aceitável (Δμ, saturação ou pixels fora da faixa real) | % dos cortes | 0% | 46% [30%, 62%] | 8% [0%, 19%] | melhorou (+37.8 pp) |
| Wavelet LL (média local, 4×4 px) | diferença do brilho médio vista pela wavelet, em fração da régua | \|sintética − original\| | 0.000 | 0.115 [0.079, 0.157] | 0.054 [0.021, 0.078] | mudança pequena (+0.032) |
| Mapa: brilho por bloco (16 px) | área do pulmão em que o bloco clareou ou escureceu além da tolerância | % do pulmão | 0% | n/a | 0% [0%, 5%] | n/a |

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

| Técnica | O que mede | Unidade | Ideal | Dgen | Dsr | Dgen → Dsr |
|---|---|---|---|---|---|---|
| Wavelet 2 px (textura fina) | detalhe entre pixels a ~2 px: bordas finas, ruído, pequenos vasos | % da original | 100% | 58% [44%, 84%] | 54% [49%, 62%] | sem diferença clara (+5.1 pp) |
| Wavelet 4 px (textura grossa) | detalhe em estruturas de ~4 px: vasos médios, septos | % da original | 100% | 60% [44%, 72%] | 66% [60%, 70%] | melhorou (+11.3 pp) |
| Wavelet 2 px ÷ 4 px (tipo de perda) | abaixo de 1: perdeu mais o fino (borrou); acima de 1: sobrou fino demais (ruído) | razão | 1.00 | 0.98 [0.91, 1.04] | 0.83 [0.82, 0.86] | piorou (-0.09) |
| GLCM contraste (vizinhos a 1 px) | quanto o tom muda de um pixel para o vizinho | % da original | 100% | 61% [50%, 89%] | 60% [54%, 65%] | mudança pequena (+4.2 pp) |
| GLCM energia (uniformidade) | quanto os pares de tons vizinhos se repetem; sobe quando a imagem fica lisa | % da original | 100% | 99% [64%, 199%] | 133% [112%, 153%] | melhorou (+33.6 pp) |
| Veredito Wavelet | cortes com textura alterada (borrou, ruído, perdeu ou ganhou contraste, estourou) | % dos cortes | 0% | 86% [76%, 97%] | 86% [73%, 97%] | sem diferença clara (+0.0 pp) |
| Mapa: textura por bloco (16 px) | área do pulmão em que o bloco tem textura alterada | % do pulmão | 0% | n/a | 93% [87%, 94%] | n/a |
| Mapa: correlação espacial | a textura está nos mesmos lugares da original? (1 = sim) | correlação | 1.00 | n/a | 0.87 [0.84, 0.89] | n/a |

| Contagem de cortes | Dgen | Dsr |
|---|---|---|
| borrou | 10 | 27 |
| estourou | 3 | 0 |
| ganhou contraste | 2 | 0 |
| perdeu contraste | 14 | 5 |
| ruído | 3 | 0 |
| textura preservada | 5 | 5 |

Ganho = quanto, corte a corte, a versão nova chegou mais perto do ideal (mediana; em taxas, pontos percentuais). "melhorou" ou "piorou" só quando o intervalo de 95% não cruza o zero e a mudança passa do mínimo relevante (5 pp nas porcentagens e taxas; 0,05 na razão, na correlação e no |Δμ|); fora do zero mas abaixo do mínimo: "mudança pequena". n/a: o mapa compara cada bloco com o mesmo bloco da original; numa versão só máscara (outro pulmão no mesmo contorno) esse par de blocos não existe.
