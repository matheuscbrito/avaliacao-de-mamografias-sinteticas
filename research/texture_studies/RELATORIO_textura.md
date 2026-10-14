# Avaliação de textura — pares real/sintético do `gen-fid`

225 pares de ROIs de mamografia (CBIS-DDSM real × difusão latente), 512×512.

| | |
|---|---|
| Repositório | `/Users/bringel/ARIA/UPenn/avaliacao-de-mamografias-sinteticas` |
| Commit base | `f9f8b29` |
| Execução | 2026-09-04 |
| Testes | 32 passando |

---

## 1. Objetivo

Duas perguntas:

1. **Quais métricas de textura são capazes de discriminar real de sintético** — e portanto quais servem para validar o gerador?
2. **Qual é a consistência entre real e sintético**, e ela é parecida entre os conjuntos (batches, Calc/Mass, Test/Training)?

A segunda só tem sentido depois da primeira: uma métrica que não discrimina nada não pode atestar consistência nenhuma.

---

## 2. O princípio metodológico

Uma métrica que não separa o real de uma imagem **sabidamente ruim** é cega, e o número dela não valida nada. Por isso toda métrica é testada contra degradações conhecidas antes de ter o direito de opinar sobre o modelo.

"Ruim", porém, tem direção — e é por isso que há **quatro** controles:

| controle | degradação | pergunta que responde |
|---|---|---|
| `bicubic_2`, `bicubic_4` | suavização (perda de alta frequência) | a métrica enxerga borrão? |
| `real_noise` | excesso de alta frequência, calibrado para igualar o sintético (razão 2,48 vs 2,47 em λ=4 px) | a métrica enxerga ruído injetado? E — mais importante — separar `syn` de `real_noise` mede **estrutura**, não quantidade de energia |
| `real_q1` × `real_q2` | nenhuma: dois quadrantes disjuntos da **mesma** ROI real | qual é o piso de ruído da métrica? |
| `*_lp` (λ > 8 px) | — | o desvio sobrevive sem a escala fina? |

**Critério.** Uma métrica só opina se (i) separa o real de algum degradado (AUC ≥ 0,80) e (ii) **não** separa dois pedaços do mesmo tecido real. O veredito sai da comparação entre banda completa e passa-baixa.

> O plano original previa apenas o controle bicúbico. Os dados mostraram que o modelo falha na direção **oposta** à do bicúbico (excesso, não falta, de alta frequência) — um controle só de suavização deixaria quase toda métrica "passar" de graça.

---

## 3. O que os dados exigiram antes de qualquer medição

### 3.1 O sintético não está na escala do real

Medido: `syn ≈ 0,38 × real + 0,34`. O modelo devolve a imagem com a faixa dinâmica comprimida a ~40 %. Sem alinhamento, GLCM com bins globais, SSIM, PSNR e gradiente mediriam offset de intensidade.

O alinhamento usa **casamento de momentos**, não mínimos quadrados: LS é atenuante (`var(syn_al) = var(real)/corr²`) e sua inflação depende da correlação do par — medido **Pearson(corr, inflação) = −0,87**. Como os batches diferem em correlação média (0,915 / 0,865 / 0,927), LS confundiria qualidade de pareamento com excesso de potência, justamente na comparação entre conjuntos que a pergunta 2 exige.

### 3.2 O real é 12 bits; o sintético é float contínuo

Real: grade 1/4095, ~1 300 valores distintos por ROI. Sintético: ~247 000 (quase todo pixel único). Sem requantizar, LBP e gradiente separariam os conjuntos pela estrutura de **empates**, não pela textura.

### 3.3 O real também não é limpo

~25 % das ROIs reais têm blocagem JPEG de 8 px (CBIS-DDSM distribuído comprimido). Flagadas, nunca excluídas; análise de sensibilidade no notebook 03.

### 3.4 Pixel spacing desconhecido

Tudo roda em ciclos/pixel. β **não** é comparável com a literatura em mm⁻¹ (a inclinação é invariante à unidade, a banda de ajuste não).

---

## 4. Defeitos encontrados durante a implementação

Cinco, todos agora travados por teste. Nenhum estava previsto no plano.

### 4.1 `band_fractions` incluía os cantos anisotrópicos da grade

21,5 % dos modos (|f| > 0,5) só existem nas diagonais e entravam no denominador. Inofensivo em campos β=3 (0,05 % da potência), mas **22 %** em conteúdo de alta frequência — exatamente o caso do sintético.

### 4.2 `harmonic_excess` descartava o bin de Nyquist silenciosamente

Ao investigar, descobri que **minha própria sondagem inicial havia lido o artefato errado**: eu tomava `max` sobre os harmônicos de `size=4`, e o máximo vinha do bin de Nyquist. Reportei "grade de 4 px" no plano onde estava escrito "alternância de 2 px".

Medido com as estatísticas separadas (75 pares):

| assinatura | real | sintético | AUC |
|---|---|---|---|
| **Nyquist (período 2 px)** | 0,67 | **2,71** | **1,000** |
| período 4 px | 0,58 | 0,47 | 0,614 |
| período 8 px (JPEG) | 0,72 | 0,69 | 0,600 |

O artefato é xadrez de upsampling 2× (convolução transposta / pixel-shuffle), confirmado pela autocorrelação do perfil de |gradiente|: lag 1 = −0,89, lag 2 = +0,91.

### 4.3 Alinhamento por mínimos quadrados acoplado à qualidade do par

Ver §3.1. Trocado por casamento de momentos.

### 4.4 Doze features da GLCM eram duplicatas exatas

`graycomatrix` arredonda o deslocamento para pixels inteiros: `d=2` a 45° vira `round(±1,414) = ±1`, o mesmo offset de `d=1`. O rótulo `d2_a45` é enganoso (a separação real é 1,41 px, não 2). Contá-las como testes distintos inflava a correção de Benjamini-Hochberg.

### 4.5 Métricas pareadas e detectores de artefato fora da classificação por banda

As pareadas (`*_vs_real`) são degeneradas no próprio real (SSIM = 1, PSNR = ∞) e teriam AUC ≈ 1 automaticamente. Os detectores de artefato são razões pico/fundo e sobrevivem ao passa-baixa por construção. Ambos ganharam categoria própria.

---

## 5. Resultados

### 5.1 Classificação das 173 métricas

| veredito | n | leitura |
|---|---|---|
| `cega` | 82 | não separa o real de nenhum degradado — reportadas, mas sem direito de opinar |
| `falha_escala_tecido` | 42 | desvio sobrevive ao passa-baixa |
| `falha_escala_fina` | 19 | desvio só na escala fina; tecido consistente |
| `inconclusiva` | 18 | AUC entre 0,6 e 0,8 |
| `modelo_passa` | 8 | indistinguível do real nesse eixo |
| `artefato_de_geracao` | 4 | detectores de xadrez 2 px / blocagem JPEG |

**Zero métricas instáveis**: `auc_floor` ≈ 0,50 em todas. O piso de quadrantes validou a medição inteira.

### 5.2 Redistribuição espectral — o achado central

Razão de potência sintético/real, média geométrica, IC 95 % por bootstrap agrupado por paciente:

| λ (px) | sintético | bicúbico 2× | bicúbico 4× | real+ruído | **piso** |
|---|---|---|---|---|---|
| 512–64 | 1,06–1,19 | 1,00 | 0,98 | 1,00 | **1,05–1,12** |
| 32 | **0,78** [0,70–0,86] | 0,99 | 0,92 | 1,01 | 1,07 |
| 16 | 1,34 | 0,97 | 0,70 | 1,08 | 1,05 |
| 8 | 2,16 | 0,88 | 0,16 | 1,34 | 1,00 |
| 4 | 2,47 | 0,39 | 0,00 | 2,48 | 0,97 |

**O piso derrubou um achado:** o excesso de 6–19 % em escala grande cai dentro do erro de medição e não é resultado. O que sobrevive é uma **redistribuição** — o modelo tira 22 % da potência em λ ≈ 32 px e põe até 2,5× abaixo de 16 px.

### 5.3 O artefato de 2 px separa sozinho

`nyquist_excess`: AUC **1,000** contra o sintético, com as faixas nem se tocando (real p90 = 1,40; sintético p10 = 2,47). É artefato de geração, não textura — daí a categoria própria.

### 5.4 O desvio é padrão, não ruído

`auc_syn_vs_noise` ≈ 1,0 para LBP e desvios angulares da GLCM. Como `real_noise` foi calibrado para ter a **mesma** energia de alta frequência do sintético, separar os dois não mede quantidade — mede estrutura. O que o modelo adiciona é espacialmente organizado.

### 5.5 Microcalcificações — o achado clinicamente mais grave

Top-hat r = 5 px, limiar mediana + 6·MAD do real, área ≥ 3 px. 186 das 225 ROIs reais têm ≥ 1 objeto (mediana 2).

| conjunto | recall | ROIs c/ recall = 0 | nº objetos (mediana) |
|---|---|---|---|
| **sintético** | **0,24** | **57 %** | 12 |
| bicúbico 2× | 0,65 | 12 % | 2 |
| bicúbico 4× | 0,12 | 67 % | 0 |
| real + ruído | 0,97 | 1 % | 10 |
| *real* | — | — | *2* |

O difusor apaga microcalcificações numa taxa comparável à de um **downsample 4×**, enquanto um bicúbico 2× ingênuo preserva 0,65.

O controle de ruído impediu um exagero: o sintético mostra 12 objetos contra 2 do real, mas `real_noise` também infla para 10 — a **contagem não discrimina** (é sensibilidade do detector sob alta frequência). O **recall** discrimina, porque o ruído não o degrada.

### 5.6 Consistência entre os conjuntos (pergunta 2)

Kruskal-Wallis dos deltas pareados, com Benjamini-Hochberg, sobre as 91 métricas úteis:

| estrato | métricas divergentes |
|---|---|
| batch | 10 |
| Calc/Mass | 2 |
| Test/Training | **0** |

As afetadas por batch são justamente as de **artefato** (`harmonic_8px`, `nyquist_excess`, `rolloff_decades`) — os batches diferem no artefato de geração, não na textura do tecido.

### 5.7 Por que p-valor não serve aqui

167 de 173 métricas têm viés significativo após BH. Com 225 pares pareados, o teste responde "o desvio é exatamente zero?" — e a resposta é sempre não.

Contra o piso de medição, apenas **51 de 173** têm desvio real (`bias_vs_floor` > 1). Essa é a coluna que discrimina.

### 5.8 Fidelidade estrutural — com a ressalva

| métrica | sintético | bicúbico 2× | bicúbico 4× | real+ruído |
|---|---|---|---|---|
| SSIM | 0,475 | **0,924** | 0,820 | 0,738 |
| PSNR (dB) | 25,18 | **39,14** | 35,12 | 31,74 |
| LBP JS (P8R1) | **0,072** | 0,304 | 0,578 | 0,190 |

O bicúbico **vence** o difusor em SSIM e PSNR. Isso é o esperado, não uma surpresa: essas métricas premiam borrão, e um modelo que devolvesse a média de todas as texturas plausíveis maximizaria PSNR. São referência de fidelidade estrutural, **não** evidência de textura.

Nuance: na distância de Jensen-Shannon dos histogramas LBP o sintético é o **mais próximo** do real (0,072). Ou seja — ele produz estatística de microtextura plausível, mas não reproduz a estrutura naquele local e destrói os objetos pequenos.

---

## 6. Onde estão os resultados

Raiz: `/Users/bringel/ARIA/UPenn/avaliacao-de-mamografias-sinteticas`

### 6.1 Resultados das métricas — `outputs/gen-fid/` (fora do git)

| arquivo | tamanho | conteúdo |
|---|---|---|
| **`decision_table.csv`** | 87 KB | **A tabela principal.** 173 métricas × 27 colunas |
| `metrics.parquet` | 4,9 MB | 594 450 linhas — todos os escalares, formato longo |
| `profiles.parquet` | 5,9 MB | 691 200 linhas — perfis espectrais radiais completos |
| `qc.parquet` | 53 KB | 6 075 linhas — 27 flags por par |
| `manifest.csv` | 72 KB | 225 pares, pareamento materializado |
| `derived_constants.json` | — | constantes globais derivadas do conjunto real |
| `sidecar_0{0,1,2}_*.json` | — | proveniência por rodada |

**`decision_table.csv`** — colunas-chave: `max_auc_control`, `auc_floor`, `auc_syn`, `auc_syn_lp`, `auc_syn_vs_noise`, `bias_median` + IC, `bias_vs_floor`, `disp_ratio`, `p_value`, `p_value_bh`, `p_heterogeneity_{batch,abnormality,split}_bh`, `noise_floor`, `verdict`.

**`metrics.parquet`** — schema `pair_id, set, metric, value, config_hash, timestamp`. 14 conjuntos: `real`, `syn`, `bicubic_2`, `bicubic_4`, `real_noise`, `real_q1`, `real_q2` + a versão `_lp` de cada um.

**`profiles.parquet`** — schema `pair_id, set, radial_bin, freq_cpx, power, config_hash`. Guarda o perfil **inteiro**, não só β: é o que permite mudar a banda de ajuste depois sem recalcular nada.

**`qc.parquet`** — 27 flags por par: `corr`, `affine_a`, `affine_a_ls`, `zero_frac_real`, `harmonic_{4,8}px_{real,syn}`, `nyquist_excess_{real,syn}`, `distinct_{real,syn}`, `value_step_real`, `near_constant`… **Nenhuma linha foi removida por QC** — QC é coluna de flag.

**`manifest.csv`** — colunas `pair_id, real_path, syn_path, batch, abnormality, split, patient_id, image_id, laterality, view, roi_index`. Pareamento materializado, nunca `sorted()` de dois diretórios.

**`derived_constants.json`** — derivadas do conjunto **real** e aplicadas identicamente a todos: `noise_sigma = 0,01259`, `gradient_threshold = 0,09572`, `glcm_quant_lo/hi = [0,0000; 0,8228]`.

**Sidecars** — `sidecar_00_inventory.json` (hash `9030ba1f2917`), `sidecar_01_spectrum.json` (`cab42bff5d31`), `sidecar_02_metrics.json` (`cbb33a8227c3`). Cada um com parâmetros, `config_hash`, git HEAD (`f9f8b29`), versões das libs, `physical_units=false`, `bicubic_origin="inferido"`, `input_is_real_roi="não confirmado"`.

### 6.2 Notebooks executados — 38 células, 0 erros

| notebook | tamanho | conteúdo |
|---|---|---|
| [`notebooks/00_inventory.ipynb`](notebooks/00_inventory.ipynb) | 1,7 MB | QC, painéis visuais, geração dos controles |
| [`notebooks/01_spectrum_ratio.ipynb`](notebooks/01_spectrum_ratio.ipynb) | 349 KB | curvas de razão + IC, β, estratificação |
| [`notebooks/02_texture_metrics.ipynb`](notebooks/02_texture_metrics.ipynb) | 294 KB | famílias de métricas, top-hat, SSIM/PSNR |
| [`notebooks/03_eda_decision.ipynb`](notebooks/03_eda_decision.ipynb) | 184 KB | tabela de decisão, EDA final, sensibilidade |

Os fontes são os `.py` de mesmo nome (formato percent, diff legível); os `.ipynb` são gerados por [`notebooks/build_notebooks.py`](notebooks/build_notebooks.py).

### 6.3 Código — 2 682 linhas

| arquivo | linhas | conteúdo |
|---|---|---|
| [`src/texture/store.py`](src/texture/store.py) | 139 | `Config`, `cfg_hash`, `upsert_parquet` |
| [`src/texture/spectrum.py`](src/texture/spectrum.py) | 234 | detrend, Hann, PSD radial, β, bandas, harmônicos, `nyquist_excess` |
| [`src/texture/io.py`](src/texture/io.py) | 212 | manifesto, alinhamento, requantização, controles |
| [`src/texture/metrics.py`](src/texture/metrics.py) | 208 | gradiente, GLCM, LBP, lacunaridade, top-hat, SSIM/PSNR |
| [`src/texture/stats.py`](src/texture/stats.py) | 342 | AUC, bootstrap por paciente, viés, dispersão, BH, dedup, veredito |
| [`src/texture/pipeline.py`](src/texture/pipeline.py) | 220 | estágios 00/01/02 |
| [`tests/test_estimators.py`](tests/test_estimators.py) | 416 | 32 testes |
| [`config.yaml`](config.yaml) | — | parâmetros; hash por subárvore |

Boa parte de `spectrum.py` e `metrics.py` foi **extraída** do antigo `extracao_metricas_textura.ipynb` (bancada para INbreast, removida do repo desde então), que já tinha esse código validado — não reescrita. O que veio de lá: `power_spectrum_2d`, `radial_average`, `fit_power_law`, `quantize`, `glcm_features`, `upsert_parquet`, `cfg_hash` e os testes de recuperação de β.

### 6.4 Como reproduzir

```bash
uv pip install -r requirements.txt
python -m pytest tests/                            # 32 testes
PYTHONPATH=src python -m texture.pipeline all      # ~35 min
python notebooks/build_notebooks.py
cd notebooks && PYTHONPATH=../src jupyter nbconvert \
    --to notebook --execute --inplace 0*.ipynb
```

Verificação de sanidade — se estes números não saírem, algo quebrou na normalização:

- `corr` mediana **0,933**
- `affine_a_ls` mediana ≈ **0,38**
- `nyquist_excess` sintético mediana ≈ **2,7** decadas
- razão espectral em λ = 4 px ≈ **2,5**
- `auc_floor` ≈ **0,50** em toda métrica

---

## 7. Ressalvas

Acompanham qualquer uso destes números:

- **Pixel spacing desconhecido.** Tudo em ciclos/pixel; β não comparável com a literatura de mamografia em mm⁻¹.
- **O fator `k` do controle bicúbico é inferido** — o gerador não salvou as imagens de baixa resolução. O controle demonstra sensibilidade das métricas; **não** reproduz a entrada real do modelo.
- **~25 % das ROIs reais têm blocagem JPEG.** Ao excluí-las, 20 métricas mudam de veredito — todas entre categorias vizinhas (`inconclusiva` ↔ `falha_*`), nenhuma invertendo "passa" para "falha de tecido". A conclusão central é robusta; as fronteiras não.
- **SSIM/PSNR premiam borrão.** Não são evidência de textura.
- **225 pares para 202 pacientes**: 6 imagens contribuem com mais de uma ROI. Todo bootstrap é agrupado por paciente.

---

## 8. Perguntas para o time do dataset (`mamografIA_upscaler`)

Em ordem de importância:

1. **Qual é exatamente a entrada do modelo?** O próprio `_roi.tiff` 512×512, ou existe uma versão reduzida? Se reduz: qual fator e com ou sem anti-aliasing antes de decimar?
   → Sem isso não sabemos nem se isto é super-resolução, e o `k` do controle bicúbico continua inferido.

2. **A saída é reescalada para a faixa do real?** Medido `syn ≈ 0,38·real + 0,34` — faixa dinâmica comprimida a ~40 %.
   → Se for bug de desnormalização, é o conserto mais barato do pipeline. Se é intencional, precisamos saber para não "corrigir".

3. **De onde vem a alternância de 2 px?** Assinatura de convolução transposta / pixel-shuffle no upsampling do decodificador AEKL.
   → Separa real de sintético com AUC 1,000 e é provavelmente o defeito mais fácil de corrigir (upsample+conv em vez de conv transposta).

4. **As ROIs reais vieram do DICOM ou da versão JPEG?** Recorte nativo ou redimensionado? Qual o pixel spacing?
   → ~25 % da **referência** está comprimida, e sem spacing β fica sem comparação com a literatura.

5. **A função de perda penaliza perda de estruturas pequenas e de alto contraste?** Uma perda puramente L1/L2 ou perceptual tende exatamente ao comportamento observado (recall 0,24 em microcalcificações).

---

## 9. Conclusão

**Sobre a pergunta 1 — quais métricas validam textura.** As de GLCM em `d=1` (contraste, homogeneidade, dissimilaridade: AUC ~0,90 contra o sintético, ~0,997 contra os controles), `grad_median` (0,904) e as frações espectrais por banda. **82 das 173 são cegas** e não devem ser usadas como evidência — inclusive várias que apareceriam bem numa tabela ingênua de "média ± desvio".

**Sobre a pergunta 2 — consistência entre conjuntos.** A consistência real↔sintético é **estável** entre Test/Training (0 métricas divergentes) e quase estável entre Calc/Mass (2). Diverge por batch em 10 métricas, mas essas são detectores de **artefato**, não de textura: os batches diferem em quanto artefato de geração carregam.

**Sobre o modelo.** Não é um problema de suavização, como se esperava de um super-resolvedor. O modelo **redistribui** textura — tira de escala média, injeta em escala fina — carrega um xadrez de 2 px que o denuncia sozinho, e apaga microcalcificações pior que uma interpolação bicúbica ingênua.

O contraste mais útil para o time: em SSIM o difusor faz 0,475 e um bicúbico 2× faz 0,924; em recall de microcalcificações, 0,24 contra 0,65. **Nos dois eixos que mais importam clinicamente, o modelo está atrás de uma interpolação clássica** — enquanto produz estatística de microtextura (LBP) mais próxima do real que qualquer controle. Ele acerta a *aparência* da textura e erra a *estrutura*.

---

*Ver também [`README_textura.md`](README_textura.md) — como rodar, o princípio dos controles e as ressalvas em formato curto.*
