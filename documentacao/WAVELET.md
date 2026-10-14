# Wavelet no framework de textura pulmonar

Este documento explica **o que fazemos com a wavelet**, de ponta a ponta: a
região medida, o cálculo, as regras de decisão, o eixo de densidade, o
diagnóstico do tipo de defeito e o mapa por região. Cada regra aponta para o
arquivo onde ela está no código. Para rodar os scripts, veja
[`wavelet/README.md`](../wavelet/README.md). Para o framework completo (GLCM,
GLRLM, espectro), veja [`FRAMEWORK_TEXTURA_CT.md`](FRAMEWORK_TEXTURA_CT.md).

## 1. Em uma página

**Pergunta:** a CT pulmonar sintética continua parecida com a original que a
gerou, em brilho e em textura?

**Divisão de trabalho entre os eixos:**

| Eixo | Responde | Onde |
|---|---|---|
| **Densidade (brilho)** | *Se* algo mudou: o pulmão ficou claro ou escuro demais? | `density_features.py` |
| **Wavelet** | *O que* aconteceu com a textura e em que escala (borrou, perdeu contraste, estourou, ruído) | `wavelet_features_fullimage.py`, `wavelet_diagnostico.py` |
| **Mapa por região** | *Onde* o defeito está, e se a textura está nos mesmos lugares da original | `mapa_regioes.py` |
| GLCM, NEx | Outras famílias de textura (GLCM em `../glcm`, NEx em `nyquist_excess.py`) | — |

**Três fatos que explicam o desenho:**

1. A wavelet de detalhe (energia e entropia) **não vê brilho**. Quem vê é a
   banda de aproximação, a LL. Por isso existe um eixo de densidade separado.
2. A energia média do pulmão **dilui defeitos locais**. Por isso existe o mapa
   por região.
3. Todas as comparações são **da sintética contra a própria original**. Há
   cortes reais legitimamente escuros ou lisos; a sintética que os acompanha
   está certa.

**Dataset padrão:** `dados/Dsr`. As máscaras vêm de `dados/Dgen`, porque o Dsr
não traz máscaras (as originais são idênticas nos dois). O Dgen entra só como
fonte de exemplos quebrados (branco, estourado, borrado).

## 2. Fluxo completo

```text
original + sintética + máscara
        │
        ▼
validação do caso ─────────────► descartados (com o motivo)
        │ (37 de 100 cortes passam)
        ▼
máscara erodida (raio 2 px)  =  ROI pulmonar
        │
        ├─► faixa de referência P1–P99 (pixels do pulmão de todas as originais)
        │
        ├─► DENSIDADE: mediana e frações fora da faixa      → density_status
        │
        ├─► WAVELET (Haar, 2 níveis, imagem inteira)
        │       ├─ energia e entropia dos coeficientes dentro do pulmão
        │       │        → referência das originais → classificação do par
        │       └─ LL, energia fina, energia grossa, razão fina ÷ grossa
        │                → veredito de textura (borrou, perdeu contraste…)
        │
        └─► MAPA POR REGIÃO: as mesmas leituras, bloco a bloco (16 px)
                → % de área por veredito, área mais clara, correlação espacial
```

## 3. Entrada e validação dos casos

Cada caso tem três arquivos: `caso.tiff` (original), `caso_generate.tiff`
(sintética) e `caso_mask.tiff` (máscara). Arquivos: `load_data.py`,
`validate_cases.py`.

Um corte só entra se passar **todas** estas regras (os motivos de descarte
ficam em `case_validation.csv` / `casos_invalidos.csv`):

| Regra | Valor |
|---|---|
| Os três arquivos existem e carregam | — |
| Original, sintética e máscara têm o mesmo tamanho | — |
| Não há valores não finitos (NaN, inf) | — |
| A máscara tem pelo menos **2 componentes** pulmonares | cada um com ≥ 800 px |
| Área pulmonar na imagem | entre **3% e 35%** |
| Depois da erosão (raio 2 px) sobram pelo menos | 800 px |

É um filtro operacional para ficar só com cortes centrais, com os dois
pulmões. Não é revisão clínica. No dataset atual, **37 dos 100 cortes** passam;
os outros 63 são descartados com motivo.

A erosão tira a borda do pulmão, onde a transição para o tecido de fora é
intensa. A **mesma máscara** vale para original e sintética, porque as duas
estão alinhadas (a sintética foi gerada a partir da original).

## 4. A faixa de referência P1–P99

Antes de qualquer cálculo, juntamos os pixels pulmonares (dentro da máscara) de
**todas as originais aceitas** e pegamos o percentil 1 e o 99. Essa é a faixa
normal de um pulmão real (`reference_interval` em `density_features.py`).

Ela é usada em dois lugares:

- como **corte da wavelet**: a imagem é limitada a [P1, P99] e normalizada para
  0–1 antes da decomposição. Isso deixa todas as imagens na mesma régua;
- como **régua da densidade**: "claro demais" quer dizer "acima do P99 real".

> **Atenção:** o corte é um passo antes da wavelet, e é ele quem causa o pouco
> efeito que o brilho tem na energia (veja a seção 7). A wavelet em si é
> insensível a brilho.

## 5. Como a wavelet é calculada

Arquivo: `wavelet_features_fullimage.py` (método adotado).

1. Limita e normaliza a imagem inteira para 0–1 (seção 4).
2. Decompõe **uma vez** com a DWT 2D: família **Haar**, **2 níveis**, modo
   `periodization`. Em uma imagem 512×512, a sub-banda do nível fino tem
   256×256 e a do nível grosso tem 128×128.
3. Reduz a máscara ao tamanho de cada sub-banda: cada coeficiente cobre um
   bloco de 2×2 px (fino) ou 4×4 px (grosso), e a máscara vira a fração de
   pulmão em cada bloco.
4. **Só entram na conta os coeficientes cujo bloco tem ≥ 90% de pulmão**
   (`min_tissue = 0,9`). A borda fica de fora.
5. Em cada sub-banda de detalhe (LH, HL, HH) mede:
   - **Energia:** média dos coeficientes ao quadrado. Cresce com estrutura fina
     e com contraste.
   - **Entropia:** entropia de Shannon da distribuição de `|coeficiente|`,
     dividida por `log2(N)`. A divisão é necessária: sem ela a entropia só
     mediria o tamanho da região.
6. `energy` e `entropy` finais são a média de todas as sub-bandas de detalhe.
   O CSV guarda também cada sub-banda (`energy_L1_LH`, `entropy_L2_HH`…) e a
   cobertura (`tissue_coverage`).

### Por que imagem inteira e não janelas (patches)

O método antigo (`wavelet_features.py`) cortava janelas 16×16 e decompunha cada
uma. Foi mantido, sem alteração, só para comparação
(`compare_patch_vs_fullimage.py`, relatório em
`wavelet/outputs/patch_vs_fullimage/REPORT.md`, pasta ignorada pelo git). O
método de imagem inteira usa a mesma regra de 90% de tecido, mas decompõe a
imagem toda de uma vez e seleciona os coeficientes pela máscara.

O modo `periodization` é o único em que cada sub-banda mede exatamente
`N/2^nível`. Isso mantém a correspondência coeficiente ↔ pixel simples. Com
Haar não faz diferença; com filtros mais longos (db4) o modo padrão muda a
energia em cerca de 14%.

### Atenção com os nomes dos níveis

No código e nos CSVs, **`L1` é o nível mais grosso (4 px) e `L2` o mais fino
(2 px)**, o inverso da convenção usual. Vem da ordem de `pywt.wavedec2`. Nos
textos e figuras usamos "fina (2 px)" e "grossa (4 px)" para não confundir.
Em `compare_generations.py`, a escala `e1` já é a mais fina (conversão feita
por `scale_keys`).

## 6. Classificação do par (a decisão inicial)

Arquivos: `build_reference.py`, `compare_images.py`.

**Referência.** Com energia e entropia das **originais aceitas**, calcula-se
média, mediana, desvio, Q1, Q3, **IQR** e limites de **Tukey** (Q1 − 1,5·IQR e
Q3 + 1,5·IQR). Esses limites são a "faixa esperada" de um pulmão real.

**Duas perguntas por feature (energia e entropia):**

| Pergunta | Regra | Resultado |
|---|---|---|
| Preservou o par? | `|sintética − original| / IQR` | `close` se ≤ 1 IQR em **ambas** as features; `distant` se passar de 1 IQR em qualquer uma |
| É plausível? | A sintética está dentro dos limites de Tukey? | `within` ou `outside` (basta uma feature fora) |

**Decisão:**

| Par | Referência | Status | Significado |
|---|---|---|---|
| distante | fora | `reject` | Forte indício de falha. Fica fora do augmentation. |
| distante | dentro | `manual_review` | Mudou muito mas ainda é plausível. Revisão humana. |
| próximo | fora, e a original também está fora do mesmo lado | `approved_with_observation` | Caso raro, mas a geração preservou a característica real. |
| próximo | fora, sem a original acompanhar | `manual_review` | Rara só na sintética. |
| próximo | dentro | `approved` | Preservou e é plausível. |

O limiar de **1 IQR** é uma configuração do MVP (`--pair-iqr-threshold`), não
um valor clínico. Deve ser revisado quando houver avaliação humana.

> **Limite conhecido:** nas duas gerações a energia cai sistematicamente
> (sobra ~60% da energia original), então a maioria dos cortes passa de 1 IQR
> e vira `manual_review`. A classificação quase não distingue Dgen de Dsr
> (8 → 9 aprovados). O diagnóstico das seções 8 e 9 existe para dizer *por que*.

## 7. Eixo de densidade (o ponto cego da textura)

Arquivo: `density_features.py`.

Energia e entropia da wavelet (e GLCM, NEx) foram feitas para ignorar o nível
médio de cinza. Então um pulmão que virou "tecido mole" branco pode passar como
textura boa. Foi isso que fez o Dgen "ganhar" em cortes quebrados (078 duas
vezes, 081). O eixo de densidade olha **só o histograma** dos pixels da
máscara.

**Três medidas por imagem** (na escala da faixa P1–P99):

| Medida | O que é |
|---|---|
| `lung_median` | posição da mediana do pulmão na faixa (0 = P1, 1 = P99) |
| `frac_bright` | fração do pulmão acima do P99 |
| `frac_dark` | fração do pulmão abaixo do P1 |

**Tolerâncias**, tiradas da variação entre as originais:

- `frac_bright` / `frac_dark`: limite de Tukey superior (Q3 + 1,5·IQR) da
  fração entre as originais, ou seja, quanto pulmão fora da faixa ainda é normal;
- `lung_median`: o IQR das medianas das originais, ou seja, quanto um pulmão
  real costuma diferir de outro.

**Veredito** (sintética contra a **própria original**), do mais grave ao mais
leve:

| Status | Regra |
|---|---|
| `too_bright` / `too_dark` | A sintética tem mais pulmão fora da faixa do que a original, além da tolerância (saturação ou tecido trocado) |
| `brighter` / `darker` | Nada saiu da faixa, mas a mediana andou mais de **1 IQR** |
| `ok` | nenhuma das anteriores |

**Efeito na decisão:** a densidade **não muda** o status de textura. Quando ela
não é `ok` e a textura tinha aprovado o corte (`approved` ou
`approved_with_observation`), a coluna `*_status_with_density` manda o corte
para `manual_review`. Cortes já em `manual_review` ou `reject` ficam como
estão.

**Por que comparar com a própria original:** uma regra absoluta ("mediana acima
de X") marcava errado as originais escuras e as sintéticas que as seguiam.

**Limites** (do teste de perturbação, seção 12): um deslocamento de 10% da
faixa fica abaixo de 1 IQR e não é sinalizado; a tolerância escura é bem mais
apertada que a clara (0,5% contra 3,4% do pulmão), então o lado escuro dispara
mais cedo, inclusive com ruído forte.

## 8. Como a wavelet responde ao brilho

Script: `wavelet_brightness.py`. Resultados: `resultados/wavelet_brilho/`.

Aplicamos brilho, contraste e lavagem (pulmão misturado com branco) às 37
originais e medimos energia, entropia e a média da LL, com e sem o corte
P1–P99.

| Defeito | Energia | Entropia | LL |
|---|---|---|---|
| Brilho (soma) | não muda (a variação vem do corte P1–P99) | não muda | **sobe em linha reta** |
| Contraste ×a | muda exatamente por **a²** | não muda | — |
| Lavagem | cai (75% de mistura → 7%) | não muda | **sobe** |

Conclusões: a energia de detalhe é cega a brilho; a entropia é cega a
contraste; a **LL é o único pedaço da wavelet que mede brilho**. Nos dados
reais, os cortes quebrados do Dgen ficam longe da diagonal na LL e no meio da
nuvem na energia. Figuras prontas para slides: `figuras_report.py`.

## 9. Diagnóstico do tipo de defeito (a assinatura da wavelet)

Script: `wavelet_diagnostico.py`. Resultados: `resultados/wavelet_diagnostico/`.

A densidade avisa *que* algo mudou. A wavelet diz *o quê*, com quatro leituras
da sintética **relativas à sua original**:

| Leitura | Cálculo | Pergunta |
|---|---|---|
| **LL** | média da banda de aproximação nos blocos de pulmão, em fração da faixa P1–P99; vira **deslocamento** (gerada − original) | O brilho mudou? |
| **Energia fina** | energia na escala de 2 px, em % da original | A textura fina continua? |
| **Energia grossa** | energia na escala de 4 px, em % da original | E a textura um pouco maior? |
| **Fina ÷ grossa** | razão entre as duas | Que tipo de perda? |
| Entropia | em % da original | (ajuda pouco; veja abaixo) |

**Guia de leitura:**

| O que a wavelet mostra | O que provavelmente aconteceu |
|---|---|
| LL andou, energias ≈ iguais | Só o brilho mudou; textura preservada |
| LL parada, energias caem por igual (razão ≈ 1) | Perda de contraste |
| LL sobe e energias caem por igual | Lavagem: o pulmão foi para o branco |
| Razão < 1 (perde mais o fino) | Borramento |
| Razão > 1 com energias caindo | Estourado: os vasos viraram platô |
| Razão > 1 com a fina subindo | Ruído |

### Regra do veredito de textura

Arquivo: `densidade_wavelet.py`, função `wavelet_verdict`. Vale para o pulmão
inteiro e para cada bloco do mapa. Aplicada nesta ordem:

| Ordem | Condição | Veredito |
|---|---|---|
| 1 | fina **e** grossa entre **80% e 125%** | `textura preservada` |
| 2 | razão < 0,9 | `borrou` |
| 3 | razão > 1,1 e fina > 100% | `ruído` |
| 3 | razão > 1,1 e fina ≤ 100% | `estourou` |
| 4 | razão entre 0,9 e 1,1 e fina < 100% | `perdeu contraste` |
| 4 | razão entre 0,9 e 1,1 e fina ≥ 100% | `ganhou contraste` |

> **Estes limites (80–125% e 0,9–1,1) são uma escolha exploratória, não um
> valor validado.** Eles precisam ser revisados com avaliação humana.

### O que vimos

- No dataset padrão (Dsr): LL de −0,07 (brilho parecido), mas sobra só 54% da
  energia fina e 64% da grossa. A razão é 0,84, a assinatura de um **leve
  borramento**: entre "sem defeito" (1,0) e o borrado simulado com σ=1 (0,72).
- A **entropia quase não ajuda**: ignora contraste por construção e fica perto
  de 100% em todos os defeitos.
- A wavelet **descreve a perda, mas não decide** o que é aceitável.

### Seis jeitos de clarear

Script: `densidade_wavelet.py`. Resultados: `resultados/densidade_wavelet/`.

A densidade diz "clareou" nos seis, e a wavelet tenta separá-los:

| Defeito simulado | Densidade | Wavelet |
|---|---|---|
| Só brilho (+25%) | clareou | textura preservada |
| Lavagem (40% branco) | clareou | perdeu contraste |
| Brilho + borrado (σ=1) | clareou | borrou |
| Gama (0,6) | claro demais | dividido |
| Fundo preenchido até a mediana | **normal** | quase preservada |
| Lavagem só em metade do pulmão | clareou | dividido |

Funciona para os três primeiros. Gama e fundo preenchido não ficam claros, e a
lavagem local se dilui na média do pulmão. Isso levou ao mapa por região.

## 10. Mapa por região

Script: `mapa_regioes.py`. Resultados: `resultados/mapa_regioes/`.

O número global é a média do pulmão inteiro, então um defeito local some. O
mapa divide o pulmão em **blocos de 16 px** e lê cada bloco.

### Como o mapa é construído

1. Decompõe a imagem uma vez (Haar, 2 níveis, mesmo corte P1–P99). Isso gera
   três mapas: média da LL, energia fina e energia grossa.
2. Reduz cada mapa para a grade de blocos de 16 px (média por bloco).
3. **Um bloco só conta como pulmão se ≥ 90% dele está na máscara.**
4. Para cada bloco de pulmão, compara a gerada com **o mesmo bloco da
   original**:

| Leitura | Cálculo |
|---|---|
| Brilho | LL(gerada) − LL(original), em fração da faixa |
| Energia fina / grossa | 100 · gerada ÷ max(original, **piso**) |
| Razão | fina ÷ grossa |
| Veredito | a mesma regra da seção 9 |

O **piso** é o percentil 5 das energias dos blocos de pulmão de todas as
originais. Ele evita razões explosivas quando o bloco original é quase liso.

### O que o mapa resume por corte

| Resumo | Como |
|---|---|
| `area_<veredito>` | fração dos blocos de pulmão em cada veredito (seis colunas) |
| `area_mais_claro` / `area_mais_escuro` | fração dos blocos cujo deslocamento de LL passa de ± a tolerância da densidade (1 IQR das medianas das originais) |
| `n_blocos` | blocos de pulmão do corte |
| `correlacao_textura` | **Spearman** entre os mapas de energia fina da original e da gerada: a textura está nos mesmos lugares? |
| `correlacao_brilho` | Spearman entre os mapas de LL |

A correlação só é calculada com **pelo menos 10 blocos** de pulmão; abaixo
disso fica vazia (NaN), porque não significaria nada.

### O que a correlação espacial diz

Ela separa duas situações que o número global confunde:

- **Textura borrada, mas no lugar certo:** correlação alta. O dataset padrão
  tem mediana de **0,87**.
- **Textura inventada ou fora do lugar:** correlação baixa. Os exemplos claros
  do Dgen têm mediana de **0,32**.

| | Área mais clara | Veredito mais comum | Correlação da textura |
|---|---:|---|---:|
| Simulado: só brilho | 100% | preservada | 0,99 |
| Simulado: lavagem global | 100% | perdeu contraste | 0,91 |
| Simulado: brilho + borrado | 100% | borrou | 0,95 |
| Simulado: lavagem em metade | **54%** | preservada / perdeu contraste | **0,43** |
| Dataset padrão | 0% | borrou (61%) | 0,87 |
| Exemplos claros (Dgen) | 93% | borrou (57%) | **0,32** |

### Por que 16 px

Testamos 8, 16 e 32 px. Com 32 px sobram cerca de 9 blocos por corte e 20 dos
37 cortes ficam sem correlação. 8 e 16 px dão a mesma conclusão (todo corte do
dataset padrão tem correlação maior que todo exemplo claro). Escolhemos 16 px.
O valor de **um corte isolado** muda com o tamanho do bloco (ρ = 0,65 entre 8 e
16 px), então vale comparar **grupos**, não cravar o número de um corte.

### Leitura das figuras

`figuras/mapa_simulado.png` e `figuras/mapa_reais.png`: cada linha tem quatro
painéis.

1. original;
2. gerada;
3. **brilho por região** (vermelho = mais claro, azul = mais escuro);
4. **textura por região** (uma cor por veredito) e a correlação no título.

O contorno verde é a máscara. Dentro de regiões muito brancas, o veredito vira
"estourou" ou "borrou" porque o corte em P99 achata os pixels; nesses blocos o
brilho é a informação principal.

## 11. Comparação Dgen × Dsr (histórica)

Script: `compare_generations.py`. Dashboard: `dashboard/pages/1_Dgen_x_Dsr.py`.
Resultados: `resultados/comparacao_Dgen_Dsr_wavelet/`.

Roda o método de imagem inteira nas duas gerações, nos mesmos 37 cortes, com as
máscaras do Dgen reutilizadas **depois de confirmar que as originais são
idênticas**. O Dsr ficou mais perto da original em energia (26 de 37) e em
entropia (31 de 37), e a ordenação dos cortes se mantém (Spearman ≈ 0,8). Já a
energia do Dgen tem correlação negativa com a original (−0,43).

> **Direção atual:** não fazemos mais novas comparações Dgen × Dsr. O Dsr é o
> dataset padrão, e cada método é lido como sinal de diagnóstico sobre ele.
> Este script e a página do dashboard ficam como registro.

## 12. Teste de perturbação

Script: `perturbation_test.py`. Resultados: `resultados/teste_perturbacao_wavelet/`.

Aplica 19 defeitos conhecidos às **originais** e mede os quatro eixos (densidade,
wavelet, GLCM, NEx). A mudança de cada métrica é dividida pelo IQR entre as
originais, então todos ficam na mesma régua: |efeito| > 1 quer dizer que o
defeito mexeu mais do que um pulmão real difere de outro.

| Defeito | Densidade | Wavelet | GLCM | NEx |
|---|---|---|---|---|
| Brilho ±20% | sim | só no escuro (pelo corte em P1) | idem | não |
| Contraste | não | **sim** | **sim** | não |
| Lavagem | **sim** | **sim** | **sim** | não |
| Saturação no topo | não | **sim** | **sim** | não |
| Suavização | não | **sim** | **sim** | **o mais forte** |

Densidade e textura são **complementares**: nenhuma cobre a outra. O NEx é
imune a brilho e contraste e mede nitidez no nível do pixel.

## 13. Glossário

| Termo | Significado |
|---|---|
| **ROI** | região de interesse: o pulmão, pela máscara erodida |
| **DWT** | transformada wavelet discreta: separa a imagem em baixa e alta frequência, em várias escalas |
| **LL** | banda de aproximação: o "borrão" da imagem. Carrega o brilho. |
| **LH, HL, HH** | bandas de detalhe: bordas horizontais, verticais e diagonais |
| **Energia** | média dos coeficientes ao quadrado. Mede quanto detalhe existe. |
| **Entropia** | quão espalhada está a energia entre os coeficientes (textura irregular × poucas bordas) |
| **P1–P99** | faixa de intensidade do pulmão real; corte e régua do framework |
| **IQR** | distância entre Q3 e Q1 das originais; unidade de "quanto é muito" |
| **Tukey** | Q1 − 1,5·IQR e Q3 + 1,5·IQR: limites da faixa esperada |
| **Spearman** | correlação de ordem; aqui compara os mapas de energia da original e da gerada |
| **Piso** | percentil 5 das energias de bloco; evita razões explosivas |

## 14. Limitações e pendências

- **Amostra pequena:** 37 cortes de poucos pacientes. A referência e os limites
  saem das mesmas originais.
- **Limites exploratórios:** 80–125% de energia, razão 0,9–1,1, 1 IQR de par e
  1 IQR de mediana não foram validados com avaliação humana.
- **Defeitos simulados** são simples; um gerador real mistura vários.
- **O dashboard (`app.py` e a página Dgen × Dsr) ainda não mostra** as colunas
  de densidade, o diagnóstico nem o mapa por região. Hoje essas saídas vivem
  em `resultados/` e nas figuras.
- **A wavelet sozinha não decide:** ela descreve a perda. O limite de
  "aceitável" será definido junto com a métrica de brilho, a GLCM e o NEx.
- **Pendências:** sensibilidade ao tamanho do bloco com limites validados; juntar
  densidade, GLCM e wavelet na decisão final; adicionar o NEx.

## 15. Mapa de arquivos

| Arquivo (`wavelet/src/`) | Papel |
|---|---|
| `load_data.py` | encontra os três arquivos de cada caso e abre as imagens |
| `validate_cases.py` | regras de aceitação do corte (seção 3) |
| `wavelet_features_fullimage.py` | **método adotado:** energia e entropia na imagem inteira (seção 5) |
| `wavelet_features.py` | método antigo (patches 16×16), só para comparação |
| `build_reference.py` | referência das originais: estatísticas e Tukey |
| `compare_images.py` | classificação do par (seção 6) |
| `run_pipeline_fullimage.py` | executa o fluxo do método adotado e grava CSVs e figuras |
| `run_pipeline.py` | executa o fluxo do método antigo; também traz utilidades usadas pelos outros scripts (`write_csv`) |
| `density_features.py` | eixo de densidade (seção 7) |
| `wavelet_brightness.py` | resposta da wavelet a brilho, contraste e lavagem (seção 8) |
| `wavelet_diagnostico.py` | assinatura do defeito: LL, fina, grossa, razão (seção 9) |
| `densidade_wavelet.py` | seis jeitos de clarear e o veredito de textura |
| `mapa_regioes.py` | mapa por bloco e correlação espacial (seção 10) |
| `compare_generations.py` | comparação Dgen × Dsr (seção 11) |
| `perturbation_test.py` | teste de perturbação dos quatro eixos (seção 12) |
| `nyquist_excess.py` | estimador NEx, usado só no teste de perturbação |
| `compare_patch_vs_fullimage.py` | experimento patches × imagem inteira |
| `visualize_wavelet_example.py` | mapa de energia, só para inspeção visual |
| `figuras_report.py` | figuras para o relatório e os slides |
| `../dashboard/` | dashboard Streamlit; lê os arquivos prontos e não recalcula |
