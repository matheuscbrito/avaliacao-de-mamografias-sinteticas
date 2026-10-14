# Bancada de métricas de textura — pares real/sintético do `gen-fid`

225 pares de ROIs de mamografia (512×512): imagem real do CBIS-DDSM e a
correspondente gerada por difusão latente. A pergunta: **quais métricas de textura
conseguem discriminar real de sintético**, e o que elas dizem sobre a consistência
entre os dois — inclusive se essa consistência é a mesma entre os conjuntos.

## Como rodar

```bash
uv pip install -r requirements.txt
python -m pytest tests/                    # 31 testes de validação dos estimadores
PYTHONPATH=src python -m texture.pipeline all    # extração completa (~30 min)
python notebooks/build_notebooks.py        # gera os .ipynb a partir dos .py
jupyter lab notebooks/                     # ou: jupyter nbconvert --execute
```

Os fontes dos notebooks são os `notebooks/NN_*.py` (formato percent, diff legível);
os `.ipynb` são gerados. Saídas em `outputs/gen-fid/` (fora do git).

## O princípio que organiza tudo

Uma métrica que não separa o real de uma imagem **sabidamente ruim** é cega, e o
número dela não valida nada. Mas "ruim" tem direção, e é por isso que há quatro
controles em vez de um:

| conjunto | degradação | pergunta que responde |
|---|---|---|
| `bicubic_2`, `bicubic_4` | suavização | a métrica enxerga borrão? |
| `real_noise` | excesso de alta frequência | a métrica enxerga ruído injetado? |
| `real_q1` × `real_q2` | nenhuma (mesmo tecido) | qual é o piso de ruído da métrica? |
| `*_lp` (λ > 8 px) | — | o desvio sobrevive sem a escala fina? |

Uma métrica só opina sobre o modelo se **separa o real de algum degradado** (AUC ≥
0,80) **e não separa dois pedaços do mesmo tecido real**. O veredito sai da
comparação entre a banda completa e a passa-baixa.

## O que os dados exigiram antes de qualquer medição

Quatro fatos medidos no `gen-fid` que invalidam a abordagem ingênua:

1. **O sintético não está na escala do real** — `syn ≈ 0,38·real + 0,34`. Sem
   alinhamento, GLCM/SSIM/gradiente medem offset de intensidade, não textura.
   O alinhamento usa **casamento de momentos**, não mínimos quadrados: LS é
   atenuante (`var/corr²`) e sua inflação depende da correlação do par
   (Pearson = −0,87), o que confundiria qualidade de pareamento com excesso de
   potência justamente na comparação entre batches.
2. **O real é 12 bits (grade 1/4095); o sintético é float contínuo.** Sem
   requantizar, LBP e gradiente separam os conjuntos pela estrutura de empates.
3. **O sintético carrega alternância de período 2 px** (xadrez de upsampling 2×):
   AUC 1,000 sozinha, faixas que nem se tocam. É **artefato de geração, não
   textura** — medido sob nome próprio (`nyquist_excess`) e contornado pela banda
   passa-baixa.
4. **O modelo falha na direção oposta ao bicúbico** (excesso, não falta, de alta
   frequência). Um controle só de suavização deixaria quase toda métrica passar
   de graça.

## Estrutura

```
config.yaml              parâmetros; hash por subárvore, paths fora do hash
src/texture/
  store.py               Config, cfg_hash, upsert_parquet (parquet longo idempotente)
  spectrum.py            detrend, Hann, PSD radial, β, bandas, harmônicos, Nyquist
  io.py                  manifesto, alinhamento, requantização, controles
  metrics.py             gradiente, GLCM, LBP, lacunaridade, top-hat, SSIM/PSNR
  stats.py               AUC, bootstrap por paciente, viés, dispersão, BH, veredito
  pipeline.py            estágios 00/01/02
tests/test_estimators.py validação (β conhecido, degenerados, controles, alinhamento)
notebooks/               00 inventário · 01 espectro/razão · 02 métricas · 03 decisão
```

## Regras invioláveis (herdadas da antiga bancada INbreast, já removida do repo)

1. Toda linha gravada carrega `config_hash` (env + parâmetros da métrica).
2. **Nenhuma linha é removida por QC** — QC é coluna de flag. Exclusão silenciosa
   vira viés de seleção invisível.
3. Escrita idempotente: re-executar com o mesmo hash substitui, não duplica.
4. A lista de métricas é fixada no `config.yaml` **antes** de olhar resultados, e
   **todas** são reportadas — métrica cega é resultado, não fracasso.

## Ressalvas que acompanham qualquer uso destes números

- **Pixel spacing desconhecido.** Tudo em ciclos/pixel; β não é comparável com a
  literatura em mm⁻¹ (a inclinação é invariante à unidade, a banda de ajuste não).
- **O fator `k` do bicúbico é inferido** — o gerador não salvou as imagens de baixa
  resolução. O controle demonstra sensibilidade das métricas; não reproduz a
  entrada real do modelo.
- **~25 % das ROIs reais têm blocagem JPEG** (CBIS-DDSM distribuído comprimido).
  Flagadas, nunca excluídas; o notebook 03 roda a análise de sensibilidade.
- **SSIM/PSNR são referência de fidelidade estrutural, não evidência de textura** —
  premiam borrão. Se o bicúbico vencer o difusor neles, é o esperado.

## Perguntas em aberto para quem gerou os dados (`mamografIA_upscaler`)

1. Qual é a entrada do modelo para o `gen-fid` — o próprio `_roi.tiff` 512×512?
   Há redução interna, com qual fator e método?
2. A saída é reescalada para a faixa do real? (observado `syn ≈ 0,38·real + 0,34`)
3. De onde vem a alternância de 2 px — estágio de upsampling do decodificador AEKL?
4. As ROIs reais vêm de DICOM nativo ou da versão JPEG? Recorte nativo ou
   redimensionado? Qual o pixel spacing?
