# Referência real do NEx na ROI do MVP

**Data:** 16/10/2026
**Amostra:** 37 pares válidos (as mesmas 37 originais para Dsr e Dgen), ROI pulmonar (máscara = 1) erodida com raio 2, como no MVP.
**Script:** `frameworks/texture_evaluator/nex_referencia.py`
**Pergunta:** qual é a faixa do NEx nos pulmões reais quando ele é medido na mesma ROI que GLCM e Wavelet recebem? E onde ficam Dsr e Dgen em relação a ela?

## Por que recalcular

Os números de [`origem_assinatura_nex`](../origem_assinatura_nex/resumo.md) foram medidos na **imagem inteira** com o fundo harmonizado. Ali entram corpo, fundo e a borda entre eles. O MVP entrega a cada métrica só o **pulmão erodido**, então a faixa real precisa ser medida nessa ROI. Senão, o limite viria de outra região.

## Como o NEx é medido na ROI

O NEx precisa de um recorte retangular, e a ROI é irregular. Então, em cada imagem:

- janelas de **32 × 32 px**, passo 16 px, **inteiramente dentro** da ROI erodida (a borda da máscara contaminaria o espectro);
- NEx de cada janela; o valor da imagem é a **mediana** das janelas;
- com menos de 5 janelas, o caso recebe a flag `poucas_janelas` e continua na tabela.

Essas escolhas foram fixadas antes de olhar os resultados. Sensibilidade: a mesma medida aceitando janelas com ≥ 90 % da área na ROI (`nex_roi_90`).

Nas originais, a mediana é de **19 janelas por imagem**. Dois cortes têm menos de 5 (4 e 1 janela).

## 1. A faixa real

| Medida | Mediana | IQR | P5 – P95 |
|---|---|---|---|
| **NEx na ROI (referência)** | **0,03** | **0,23** | **−0,26 a 0,25** |
| NEx na ROI, janelas ≥ 90 % | 0,04 | 0,16 | −0,18 a 0,22 |
| NEx na imagem inteira (sem harmonizar) | 0,16 | 0,25 | −0,23 a 0,44 |

- **No pulmão, o NEx real fica em torno de zero.** Na imagem inteira, a mediana sobe para 0,16 e o P95 para 0,44. Um limite tirado da imagem inteira seria frouxo demais para o pulmão.
- A janela de 90 % dá uma faixa parecida, um pouco mais estreita. O NEx depende do tamanho e da posição da janela: qualquer limite vale só para este protocolo.

## 2. Dsr e Dgen contra a faixa

| | Dsr — ROI | Dsr — imagem inteira | Dgen — ROI | Dgen — imagem inteira |
|---|---|---|---|---|
| Mediana da sintética | 0,10 | 0,94 | **0,38** | 0,76 |
| Δ pareado (mediana) | +0,10 (0,4 IQR) | +0,86 | +0,36 (1,5 IQR) | +0,59 |
| Pares acima do real | 26/37 | 37/37 | 35/37 | 35/37 |
| AUC real × sintética | 0,66 | 0,99 | **0,93** | 0,93 |
| Sintéticas acima do P95 real | 16 % | 97 % | **65 %** | 73 % |

- **Dsr:** dentro do pulmão, a assinatura é fraca (AUC 0,66; 16 % acima do P95). Quase toda a separação da imagem inteira vem de **fora do pulmão** (corpo e fundo).
- **Dgen:** a assinatura está **dentro do pulmão** (AUC 0,93; 65 % acima do P95).
- Nenhuma sintética, em nenhuma versão, ficou abaixo do P5 real: o deslocamento é sempre para cima.
- A janela de 90 % leva às mesmas conclusões (Dsr: AUC 0,75; Dgen: 0,95).

**Como ler o Dgen:** no Dgen, a sintética é outro pulmão no mesmo contorno (modo "só máscara" do placar). O Δ pareado mistura a assinatura com a diferença de anatomia. Para ele valem a AUC e a fração acima do P95, que comparam com a faixa das originais, não com a original do par.

## O que isso muda para o framework

- A **referência do NEx no MVP** passa a ser a faixa real na ROI: mediana 0,03, IQR 0,23, P95 0,25 (`resumo.json`, chave `referencia_real.nex_roi`). Como as outras réguas, ela deve ser calibrada nas originais da execução e congelada no placar.
- O NEx na ROI responde à pergunta do framework (o pulmão sintético tem a assinatura?). Ele **não** vê a parte da assinatura do Dsr que está fora do pulmão. Se o classificador do data augmentation receber a imagem inteira, essa parte também chega a ele, e precisaria de uma leitura separada na imagem inteira.

## Limitações

- 37 cortes de poucos pacientes. A referência e as comparações saem das mesmas originais.
- Dois cortes reais têm menos de 5 janelas; o valor deles é menos estável.
- A ROI do parênquima é pequena para janelas de 32 px. Janelas menores dariam mais amostras, mas o NEx tem menos bins de frequência para separar pico de fundo.
- Aqui a imagem inteira não tem fundo harmonizado. Os números dela não são iguais aos de `origem_assinatura_nex` (100 cortes, fundo harmonizado), embora apontem na mesma direção.

## Reproduzir

Na raiz do repositório:

```bash
.venv/bin/python -m frameworks.texture_evaluator.nex_referencia \
  --version Dsr=dados/Dsr --version Dgen=dados/Dgen --mask-dir dados/Dgen \
  --output-dir resultados/nex_referencia_roi
```

Saídas: `por_caso.csv` (uma linha por par e versão) e `resumo.json` (faixa real e comparação de cada versão).
