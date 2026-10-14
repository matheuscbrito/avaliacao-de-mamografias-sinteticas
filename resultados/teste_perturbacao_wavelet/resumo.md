# Teste de perturbação controlada

**Data:** 07/10/2026
**Amostra:** 37 originais aceitas de `dados/Dgen` (as mesmas da comparação Dgen × Dsr)
**Pergunta:** cada eixo do framework reage ao defeito que diz medir, e ignora os outros?

Cada original recebeu 19 defeitos conhecidos (`figuras/exemplos_perturbacao.png`).
Para cada métrica, a figura `figuras/sensibilidade.png` mostra a mudança
mediana, dividida pelo IQR dessa métrica entre as originais. |valor| > 1 quer
dizer que o defeito mexeu mais do que um pulmão real difere de outro.

Intensidades estão em frações da faixa P1–P99 do pulmão real: "brilho +10%"
soma 10% dessa faixa à imagem inteira.

## O que cada eixo pega

| Defeito | Densidade | Wavelet | GLCM | NEx |
|---|---|---|---|---|
| Brilho ±10% | fraco (±0,7 IQR); −10% sinalizado `too_dark` em 73% | não reage | não reage | não reage |
| Brilho ±20% | sim (±1,4 IQR; 100% sinalizado) | só no escuro, pelo corte em P1 | energia +7 no escuro, pelo mesmo corte | não reage |
| Contraste ×0,5 / ×1,5 | mediana não muda | **sim** (−1,8 / +2,1) | **sim** | não reage |
| Gama 0,7 / 1,4 | **sim** (100% sinalizado) | fraco | fraco / energia +3 | não reage |
| Lavagem 25–75% | **sim** (+1,3 a +4,0; 97–100% `brighter`) | **sim** (−1,1 a −2,2) | **sim** | não reage |
| Saturação no topo (P90/P75) | não reage | **sim** (−1,3 / −1,9) | **sim** | não reage |
| Saturação no fundo (P10/P25) | não reage | entropia fraca (−0,9) | energia (+1,9 em P25) | não reage |
| Suavização σ=1 / σ=2 | não reage | **sim** (−1,2 / −1,9) | **sim** | **sim, o mais forte** (−3,4 / −4,6) |
| Ruído 2% / 5% | 5% gera `too_dark` em 41% | fraco | contraste +1,5 em 5% | fraco |

## Leitura

- **Densidade e textura são complementares.** Brilho, gama e lavagem movem a
  densidade. Contraste, saturação e suavização mexem na textura e deixam a
  mediana parada. Nenhum eixo cobre o outro.
- **A lavagem é o defeito do Dgen** (pulmão indo para tecido mole). Ela é o
  único defeito que todos os eixos, menos o NEx, percebem juntos. A densidade
  diz *o que* aconteceu (clareou); a wavelet e a GLCM dizem que a textura foi junto.
- **NEx é imune a brilho e contraste, como prometido.** Confirmado: zero em
  todas as mudanças afins. Ele reage forte só à suavização, ou seja, mede
  nitidez no nível do pixel. É o eixo certo para borramento e artefatos de
  upsampling, não para brilho.
- **A GLCM tem uma sensibilidade a brilho que não é textura.** Ela vem do corte
  fixo em P1/P99 antes de quantizar: quando o pulmão escurece, os pixels mais
  escuros colam no nível 0 e a energia sobe (+7 em −20%). Isso deve ser lido
  como efeito colateral da quantização.
- **Limites da densidade.** Um deslocamento de 10% da faixa fica abaixo de
  1 IQR e não é sinalizado para cima. A tolerância escura (0,5% do pulmão abaixo
  do P1) é bem mais apertada que a clara (3,4%), então o lado escuro dispara
  mais cedo. Ela também dispara com ruído forte, porque o ruído espalha pixels
  para fora da faixa sem mudar o brilho médio.

## Limitações

- Os defeitos são sintéticos e simples. Um gerador real mistura vários deles
  ao mesmo tempo.
- As 37 originais vêm de poucos pacientes, e a referência e os limites saem
  delas mesmas.
- O NEx foi calculado na imagem inteira, sem máscara, como no protocolo do
  repositório de origem. Então ele mistura pulmão, corpo e fundo.

## Reproduzir

A partir de `wavelet/`:

```bash
.venv/bin/python -m src.perturbation_test \
  --data-dir ../dados/Dgen \
  --output-dir ../resultados/teste_perturbacao_wavelet
```
