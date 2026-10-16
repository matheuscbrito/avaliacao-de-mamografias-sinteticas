# Placar por versão do gerador

**Data:** 15/10/2026
**Script:** `frameworks/texture_evaluator/placar.py`
**Tabela completa:** `placar.md` · **Figura:** `placar.png`

## Para que serve

O time que gera as imagens roda o placar a cada iteração do modelo e vê se a textura está chegando mais perto da real. Uma **versão** é a pasta de saída de uma iteração (por exemplo, "v1 época 50", "v2 loss nova"), gerada a partir das **mesmas originais**.

Garantias:
- **Régua fixa:** as réguas são calibradas uma vez, nas originais da primeira versão, e valem para todas. O script recusa versões cujas originais sejam diferentes.
- **Mesmos cortes:** só entram os cortes válidos em todas as versões.
- **Margem de erro:** cada número vem com um intervalo de 95% (bootstrap nos cortes).
- **Comparação justa:** cada versão é comparada com a anterior **corte a corte**. Só aparece "melhorou" ou "piorou" quando o intervalo não cruza o zero **e** a mudança passa de um mínimo relevante (5 pp, ou 0,05 na razão e no Δμ). Os mínimos são exploratórios.

## Validação

As versões abaixo foram simuladas a partir do Dsr:

| Versão simulada | Resultado esperado | Resultado |
|---|---|---|
| cópia idêntica | nada muda | "sem diferença clara" em todos os indicadores |
| borrada (σ=1) | textura piora | "piorou" em detalhe fino, detalhe grosso, razão, contraste GLCM e aprovadas; Δμ só "mudança pequena" |

## Demonstração com as duas versões reais (Dgen → Dsr)

São as duas gerações que temos das mesmas 100 originais (37 cortes válidos). Aqui servem só para mostrar como o placar se lê.

| Indicador | Ideal | Dgen | Dsr | Dgen → Dsr |
|---|---|---|---|---|
| Detalhe fino | 100% | 58% | 54% | sem diferença clara |
| Detalhe grosso | 100% | 60% | 66% | **melhorou** |
| Razão fino/grosso | 1,00 | 0,98 | 0,83 | **piorou** |
| Contraste GLCM | 100% | 61% | 60% | sem diferença clara |
| Aprovadas | 100% | 3% | 14% | **melhorou** |
| Brilho sinalizado | 0% | 46% | 8% | **melhorou** |

**Leitura para o time:**
- O Dsr resolveu o brilho: 46% → 8% dos cortes sinalizados, e nenhum rejeitado.
- A quantidade de detalhe fino continua em ~55% da original nas duas versões. **Esse é o principal problema que falta atacar.**
- O tipo de perda mudou:
  - o Dgen perdia textura por igual nas duas escalas ("perdeu contraste", razão ~1);
  - o Dsr perde mais o fino que o grosso ("borrou", razão 0,83).

  Para a próxima iteração, isso aponta para o que suaviza alta frequência (upsampling, decoder, loss).

## Cuidados

- **Razão perto de 1 não quer dizer textura boa.** Ela diz *como* a textura se perdeu (por igual ou mais no fino), não *quanto*. Leia sempre junto com o detalhe fino e o grosso.
- **Com 37 cortes, o intervalo é largo.** Mudanças menores que ~5 pp dificilmente aparecem como reais. Mais cortes válidos estreitam o intervalo. Hoje 63 de 100 ficam de fora: em 44 a máscara não tem os dois pulmões, e em 19 o pulmão ocupa menos de ~3% da imagem (cortes do topo ou da base).
- O placar mede fidelidade de textura à **original de cada par**. Ele não mede diversidade nem qualidade dos nódulos.
