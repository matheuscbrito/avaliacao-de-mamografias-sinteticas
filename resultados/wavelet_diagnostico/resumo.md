# O que a wavelet diz sobre o defeito

**Data:** 08/10/2026
**Dataset:** padrão (`dados/Dsr`), 37 cortes aceitos; o Dgen entra só como exemplo de imagem branca.

## Ideia

A métrica de brilho avisa *que* algo mudou. A wavelet ajuda a dizer *o quê*.
Ela tem quatro leituras, sempre comparando a gerada com a sua original:

| Leitura | Pergunta |
|---|---|
| **LL** | O brilho mudou? |
| **Energia fina (2 px) e grossa (4 px)** | Quanta textura sobrou? |
| **Fina ÷ grossa** | Que tipo de perda foi? |
| **Entropia** | (quase não ajuda, ver abaixo) |

Figura principal: `figuras/assinatura_wavelet.png`.

## Guia de leitura

| O que a wavelet mostra | O que provavelmente aconteceu |
|---|---|
| LL andou, energias ≈ iguais | Só o brilho mudou; a textura está preservada |
| LL parada, energias caem por igual (razão ≈ 1) | Perda de contraste |
| LL sobe e energias caem por igual | Lavagem: o pulmão foi para o branco |
| Razão < 1 (perde mais o fino) | Borramento |
| Razão > 1 com energias caindo | Estourado: os vasos viraram platô |
| Razão > 1 com a fina subindo | Ruído |

## O que vimos no dataset padrão

- **Brilho quase igual à original** (LL −0,07).
- **Perde textura:** sobra 54% da energia fina e 64% da grossa.
- **A perda é maior no fino** (razão 0,84). É a assinatura de um **leve
  borramento**, entre "sem defeito" (1,0) e o borrado simulado com σ=1 (0,72).
  O corte mais borrado (031) chega a 0,73.

O exemplo branco do Dgen (078) mostra a diferença: o LL sobe +0,64, a
assinatura de lavagem.

## Limites da wavelet

- **A entropia quase não ajuda.** Ela ignora contraste por construção e fica
  perto de 100% em todos os defeitos; só cai um pouco quando a imagem escurece.
- **Brilho só pela LL.** Energia e entropia não veem brilho; o pouco que
  mudam (≈ 90% com ±20%) vem do corte P1–P99 feito antes da wavelet.
- **A wavelet sozinha não decide nada.** Ela descreve a perda; o limite de
  "aceitável" ainda precisa ser definido junto com a métrica de brilho.
- **Amostra pequena:** 37 cortes de poucos pacientes, defeitos simulados simples.

## Reproduzir

A partir de `wavelet/`:

```bash
.venv/bin/python -m src.wavelet_diagnostico \
  --data-dir ../dados/Dsr --mask-dir ../dados/Dgen \
  --examples-dir ../dados/Dgen \
  --output-dir ../resultados/wavelet_diagnostico
```

As máscaras vêm da pasta do Dgen porque o Dsr não traz máscaras (as originais são as mesmas).
