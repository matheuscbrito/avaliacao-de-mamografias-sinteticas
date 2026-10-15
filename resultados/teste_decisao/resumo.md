# Camada de decisão: veredito final por corte

**Data:** 15/10/2026
**Dataset:** padrão (`dados/Dsr`), 37 pares válidos.
**Código:** `frameworks/texture_evaluator/decision.py` (regras) e `teste_decisao.py` (este teste).

## As regras

O MVP responde a três perguntas, nesta ordem:

| Pergunta | Leitura | O que pode decidir |
|---|---|---|
| 1. O brilho mudou? | filtro do contrato (Δμ, Sat31) | **rejeitar** ou revisar |
|  | densidade da Wavelet | só revisar |
| 2. Quanto detalhe sobrou? | contraste GLCM e energia fina Wavelet (% da original) | revisar, se as duas discordarem |
| 3. Que tipo de defeito? | veredito Wavelet (razão fino/grosso) | revisar, se não for "textura preservada" |

- **Rejeitar:** só o filtro do contrato rejeita. A textura fica registrada, mas não é interpretada.
- **Revisar:** qualquer sinal de brilho, defeito de textura ou discordância entre GLCM e Wavelet.
- **Aprovada:** nada disso aconteceu.
- Cada corte sai com os **motivos em texto** (`final_reasons`), por exemplo: "brilho (densidade): escureceu; textura: borrou (fino 43%, grosso 58%, contraste GLCM 47%)".

Defeito de textura manda para revisão, e não para rejeição, porque os limites de textura (80–125%, razão 0,9–1,1) ainda são exploratórios.

## 1. Defeitos conhecidos (simulado)

Figura: `figuras/decisao_defeitos_conhecidos.png`.

Cada original vira a própria "sintética" com um defeito conhecido e passa pelo MVP inteiro. O brilho é medido em IQR das médias pulmonares, a unidade do filtro.

| Defeito | Veredito final | Quem apontou |
|---|---|---|
| sem defeito | **37 aprovadas** | ninguém |
| brilho +0,5 | 37 aprovadas | ninguém (dentro da variação real) |
| brilho ±1,5 | 37 revisar | filtro e densidade |
| brilho +2,5 | 37 rejeitar | filtro |
| contraste ×0,5 / ×1,5 | 37 revisar | Wavelet: perdeu / ganhou contraste |
| suavização σ=1 e σ=2 | 37 revisar | Wavelet: borrou (37/37) |
| ruído 5% | 37 revisar | Wavelet: ruído (37/37) |
| ruído 2% | 20 revisar, 17 aprovadas | Wavelet: ruído (fronteira) |
| lavagem 50% | 32 rejeitar, 5 revisar | filtro (+ Wavelet: perdeu contraste) |

- **Sem falsos alarmes:** a imagem idêntica e o brilho pequeno passam em todos os cortes.
- **Cada defeito é apontado pela camada certa:** o brilho pelo brilho, e a textura pela Wavelet, com o tipo certo.
- **GLCM e Wavelet quase sempre concordam** sobre a quantidade de detalhe: no máximo 6 de 37 discordaram (ruído 2%, perto da fronteira).
- **Ruído 2% fica na fronteira:** a energia fina sobe ~27% na mediana, logo acima do limite de 125%.

## 2. Dados reais (Dsr)

| Veredito final | Cortes |
|---|---|
| aprovada | 5 |
| revisar | 32 |
| rejeitar | 0 |

- **O motivo quase sempre é textura:** 27 "borrou" e 5 "perdeu contraste". Em mediana, sobra 54% do detalhe fino.
- **Brilho só pesa em 3 cortes:**
  - 053 e 070 escureceram (filtro e densidade concordam);
  - 046 foi "escuro demais" só pela densidade.
- **Discordância GLCM × Wavelet:** 1 corte (056), na fronteira (contraste 81%, detalhe fino 71%).

## Cuidados

- **Clarear muito satura o teto da régua (vasos).** Com brilho +1,5, 18 dos 37 cortes já têm a textura lida como "perdeu contraste" ou "estourou". Isso é efeito do corte no teto, não um defeito separado. Como esses cortes já vão para revisão pelo brilho, a decisão não muda, mas o motivo de textura deve ser lido com cuidado.
- **Os limites de textura não estão validados.** "Revisar" quer dizer "olhar", não "ruim". Calibrar os limites com avaliação visual de alguém da área é o próximo passo natural.
- **NEx ainda não entra na decisão.**
