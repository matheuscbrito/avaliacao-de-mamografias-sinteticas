# Como a wavelet responde ao brilho

**Data:** 08/10/2026
**Amostra:** 37 originais aceitas (varredura) + os mesmos cortes de Dgen e Dsr (dados reais)

## Em uma frase

As bandas de detalhe da wavelet (energia e entropia) não veem brilho. Quem vê é
a banda de aproximação (LL), que hoje o framework não usa.

## O que o teste mostrou

Figura: `figuras/wavelet_resposta_brilho.png`. Linha laranja = wavelet pura (sem
corte); linha azul = como o pipeline calcula (com corte em P1–P99).

1. **Brilho:** na wavelet pura, energia e entropia ficam em exatamente 100%. A
   mudança que aparece no pipeline vem do **corte P1–P99**, não da wavelet:
   - até ±10%, quase nada muda;
   - além disso, os pixels batem no limite e a energia cai (para 62% em −30%,
     82% em +30%).
2. **Contraste:** a energia segue exatamente o fator² (×0,5 → 25%, ×2 → 400%).
   A entropia não muda: ela é cega a contraste.
3. **Lavagem** (pulmão misturado com branco, o defeito do Dgen): a energia cai
   (75% de mistura → 7%). A entropia não muda.
4. **LL:** sobe em linha reta com o brilho e com a lavagem. É o único pedaço da
   wavelet que mede brilho.

## Nos dados reais

Figura: `figuras/wavelet_ll_dados_reais.png`.

- Na LL, os cortes quebrados do Dgen (078 ×2, 081) e mais dois cortes claros
  (056, 045) ficam bem longe da diagonal: a LL subiu de 0,34 a 0,64 da faixa.
- Na energia de detalhe, esses mesmos cortes ficam no meio da nuvem. Por isso
  o Dgen "ganhava" neles.
- O Dsr fica perto da diagonal na LL (mediana −0,07, levemente mais escuro).

## Limitações

- Defeitos simulados e simples; 37 cortes de poucos pacientes.
- A LL ainda não tem limite de decisão; aqui ela só é mostrada.

## Reproduzir

A partir de `wavelet/`:

```bash
.venv/bin/python -m src.wavelet_brightness \
  --dgen-dir ../dados/Dgen --dsr-dir ../dados/Dsr \
  --output-dir ../resultados/wavelet_brilho
```
