# Mapa por região: densidade e wavelet em blocos

**Data:** 09/10/2026
**Dataset:** padrão (`dados/Dsr`), 37 cortes; exemplos claros: 12 cortes do Dgen sinalizados pela densidade.

Os números globais fazem a média do pulmão inteiro, então um defeito local se
dilui. Aqui o pulmão é dividido em blocos, e cada bloco recebe:

- **Brilho:** quanto a LL andou em relação ao mesmo bloco da original. Usa a
  mesma tolerância da densidade global (1 IQR das medianas das originais).
- **Textura:** o mesmo veredito da wavelet (borrou, perdeu contraste,
  estourou…), bloco a bloco.
- **Correlação espacial** (Spearman) entre os mapas de energia fina da original
  e da gerada: a textura está nos mesmos lugares? Ela só é calculada quando o
  corte tem pelo menos 10 blocos de pulmão.

Figuras (blocos de 16 px): `figuras/mapa_simulado.png` e `figuras/mapa_reais.png`.

## Resultados com 16 px (mediana)

| | Área mais clara | Veredito mais comum (área) | Correlação da textura |
|---|---:|---|---:|
| Simulado: só brilho | 100% | preservada (76%) | 0,99 |
| Simulado: lavagem global | 100% | perdeu contraste (80%) | 0,91 |
| Simulado: brilho + borrado | 100% | borrou (92%) | 0,95 |
| Simulado: lavagem em metade | **54%** | preservada 41% / perdeu contraste 35% | **0,43** |
| Dataset padrão | 0% | borrou (61%) | 0,87 |
| Exemplos claros (Dgen) | 93% | borrou (57%) | **0,32** |

- **O mapa resolve a lavagem local.** No número global ela ficava sem veredito
  claro; no mapa, a metade lavada aparece clara e com "perdeu contraste".
- **A correlação espacial separa "textura no lugar certo" de "textura
  inventada".** O dataset padrão borra, mas no lugar certo (0,87). Nos exemplos
  claros do Dgen, a textura não está onde estava na original (0,32).
- **Dentro de regiões muito brancas, o veredito de textura vira
  "estourou"/"borrou"**, porque o corte em P99 achata os pixels. Nesses blocos,
  o brilho é a informação principal.

## Tamanho do bloco: 8, 16 e 32 px

| | 8 px | 16 px | 32 px |
|---|---:|---:|---:|
| Blocos de pulmão por corte (mediana) | 263 | 55 | **9** |
| Cortes sem correlação (< 10 blocos) | 0 | 0 | **20 de 37** |
| Correlação, dataset padrão (mediana; mín.) | 0,83; 0,60 | 0,87; 0,58 | 0,92; 0,75* |
| Correlação, exemplos claros (mediana; máx.) | 0,17; 0,34 | 0,32; 0,55 | 0,38; 0,73* |
| Área "borrou", dataset padrão | 61% | 61% | 75% |
| Lavagem em metade: área mais clara | 50% | 54% | 55% |

\* Só os cortes com pelo menos 10 blocos (17 do dataset padrão e 5 exemplos).

- **32 px é grande demais para estes pulmões.** Sobram ~9 blocos por corte, e
  metade dos cortes fica sem correlação.
- **8 e 16 px dão a mesma conclusão.** Em ambos, todos os cortes do dataset
  padrão têm correlação maior que todos os exemplos claros. A folga entre os
  dois grupos é maior com 8 px (0,60 contra 0,34) que com 16 px (0,58 contra 0,55).
- **O valor de cada corte depende do tamanho do bloco.** A ordem dos cortes do
  dataset padrão entre 8 e 16 px tem ρ = 0,65. Então vale comparar grupos, não
  cravar o número de um corte isolado.
- **A área de "borrou" e a detecção da lavagem local quase não mudam** entre 8 e 16 px.

**Decisão:** usar só 16 px. As saídas de 8 e 32 px não foram guardadas; para
refazer, rode o comando abaixo com `--block 8` ou `--block 32`.

## Cuidados

- Os limites do veredito de textura são escolhas exploratórias.
- Nos simulados, a borda reta da lavagem cria uma faixa de "ruído" artificial.
- 37 cortes de poucos pacientes.

## Reproduzir

A partir de `wavelet/`:

```bash
.venv/bin/python -m src.mapa_regioes \
  --data-dir ../dados/Dsr --mask-dir ../dados/Dgen \
  --examples-dir ../dados/Dgen --block 16 \
  --output-dir ../resultados/mapa_regioes
```
