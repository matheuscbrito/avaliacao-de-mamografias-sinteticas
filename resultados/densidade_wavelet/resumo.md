# Densidade + wavelet: o que a wavelet denuncia quando o pulmão clareia

**Data:** 09/10/2026
**Dataset:** padrão (`dados/Dsr`), 37 cortes aceitos. Exemplos de imagens claras: cortes do Dgen que a densidade sinalizou.

A densidade diz **se** o brilho mudou. A wavelet diz **o que aconteceu com a textura** junto com o brilho.

## 1. Seis jeitos de clarear (simulado)

Figuras: `figuras/clarear_seis_jeitos.png` (tabela) e `figuras/clarear_exemplos.png` (imagens).

| Defeito | Densidade diz | Wavelet diz |
|---|---|---|
| Só brilho | clareou | textura preservada |
| Lavagem (pulmão indo pro branco) | clareou | perdeu contraste |
| Brilho + borrado | clareou | borrou |
| Gama | claro demais | dividido (12/37 preservada) |
| Fundo preenchido | **normal** | quase preservada (82%) |
| Lavagem em metade do pulmão | clareou | dividido (14/37 borrou) |

- **Funciona:** nos três primeiros, a densidade diz a mesma coisa ("clareou") e a wavelet separa os três casos.
- **Não funciona tão bem:**
  - *Gama* muda o brilho de forma não linear, e a wavelet não dá um veredito claro.
  - *Fundo preenchido* (a parte escura do pulmão sobe até a mediana) passa pelos dois eixos: a mediana não muda, e a energia cai pouco.
  - *Lavagem local*: como o número é a média do pulmão inteiro, o defeito se dilui. Isso precisa de um mapa por região, a correlação espacial que já está nos próximos passos.

## 2. Imagens geradas

Figura: `figuras/dados_reais.png`.

- **No dataset padrão, o brilho quase nunca é o problema.** A densidade diz "normal" em 34 dos 37 cortes.
- **A wavelet denuncia outra coisa: borramento.** Em 25 cortes ela diz "borrou" (sobra ~51% do detalhe fino e ~62% do grosso). Em 4, "perdeu contraste". Só 5 têm a textura preservada.
- **Nos exemplos claros do Dgen, a textura reage de jeitos diferentes:**
  - em alguns cortes ela some (078 50.0A e 078 40.0A);
  - em outros ela **aumenta** junto com o brilho (042, 059, 060: 126–174% da energia), ou seja, o gerador inventa textura enquanto clareia.

  A densidade sozinha não diferencia esses casos.

## Cuidados

- Os limites do veredito da wavelet (80–125% de energia, razão 0,9–1,1) são uma escolha exploratória, não um valor validado.
- 37 cortes de poucos pacientes; defeitos simulados simples.

## Reproduzir

A partir de `wavelet/`:

```bash
.venv/bin/python -m src.densidade_wavelet \
  --data-dir ../dados/Dsr --mask-dir ../dados/Dgen \
  --examples-dir ../dados/Dgen \
  --output-dir ../resultados/densidade_wavelet
```
