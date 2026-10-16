# Origem da assinatura NEx no Dsr

**Datas:** 25/09/2026 (reconstrução pelo AEKL) e 02/10/2026 (trajetória da amostragem)
**Amostra:** os 100 cortes do Dsr, de 3 pacientes (9, 13 e 78 cortes). As originais são as mesmas de `dados/Dsr` (hash conferido).
**Pergunta:** em que etapa do gerador aparece o deslocamento do NEx do sintético em relação ao real?
**Código e saídas originais:** repositório `gabrielbringel/nyquist-excess` (ver [Proveniência](#proveniência)).

**Resposta curta:** a assinatura aparece **na difusão/amostragem**, já na primeira previsão do denoiser. A passagem isolada pelo autoencoder (AEKL) explica cerca de **8 %** do deslocamento. O NEx, portanto, **não** deve ser lido como "artefato de upsampling do decodificador".

## Como o NEx foi medido aqui

- NEx = `max(NEx_x, NEx_y)` (`wavelet/src/nyquist_excess.py` é cópia da mesma fórmula), em décadas (log10 da razão pico/fundo perto de Nyquist).
- **Imagem inteira 512×512**, com o fundo harmonizado (`harmonize_background`: fundo fora da máscara ≥ 0,5 levado a z-score de y = 0). **Não** é a ROI pulmonar erodida do MVP: os valores daqui não são comparáveis diretamente com os do `run_mvp`. A faixa real na ROI do MVP está em [`nex_referencia_roi`](../nex_referencia_roi/resumo.md).
- A GLCM desta análise usa a régua antiga (P1–P99 dos reais, imagem inteira), não a régua Tukey k=3 do MVP.
- Unidade: corte. Os cortes de um paciente não são independentes; p-valores e ICs são descritivos.

## 1. O AEKL sozinho (25/09)

Cada corte real passou uma vez pelo AEKL do Dsr (codifica → decodifica, determinístico, pela média `z_mu`): **A(R)**. O mesmo foi feito com o sintético: **A(S)**.

| Grupo | NEx (mediana) |
|---|---|
| R — real | 0,151 |
| A(R) — real reconstruído pelo AEKL | 0,238 |
| S — Dsr | 0,714 |
| A(S) — Dsr reconstruído pelo AEKL | 0,793 |

- Deslocamento pareado só do AEKL (A(R) − R): mediana **+0,048**. Do pipeline completo (S − R): **+0,609**. **Razão 0,08.**
- AUC R × A(R) = **0,569** (quase acaso); R × S = 0,922; A(R) × S = 0,887.
- Outros estimadores de período 2 px também quase não separam R de A(R) (AUC 0,52–0,53).
- O AEKL é quase idempotente sobre o sintético (A(S) − S: +0,035).
- Leitura pré-especificada no protocolo: *o sinal aparece na difusão/amostragem, não no AEKL.*

Figura: `figuras/reconstrucao_aekl_grupos.png`.

## 2. Ao longo da amostragem (02/10)

Os 100 cortes foram gerados de novo com o mesmo LDM + AEKL do Dsr (1000 passos DDPM, noise level 1, semente 42), guardando um retrato a cada 10 % dos passos:

- **x̂_0**: a imagem final que o scheduler prevê naquele passo, decodificada;
- **x_t**: o latente ruidoso corrente, decodificado. Nos decis intermediários ele é dominado por ruído; mede a resposta do decodificador ao ruído, não a imagem. Não deve ser interpretado.

| Série | NEx (mediana) | Δ NEx pareado (mediana) | Contraste GLCM (mediana) |
|---|---|---|---|
| Real | 0,151 | — | 4,70 |
| A(R) — só AEKL | 0,238 | +0,05 (65/100 acima do real) | 4,51 |
| x̂_0 em 10 % dos passos | 0,926 | **+0,80 (91/100)** | **1,10** |
| x̂_0 em 50 % | 0,501 | | 1,12 |
| Final (100 %) | 0,690 | **+0,58 (96/100)** | 1,31 |

- **O deslocamento já está na primeira previsão do denoiser** (10 % dos passos): o NEx sobe e o contraste GLCM cai de 4,7 para 1,1. Nada disso aparece em A(R).
- Depois disso, o NEx de x̂_0 fica estável em ≈ 0,5–0,7 até o fim.
- **Por paciente**, o efeito é heterogêneo: Δ NEx final de +0,58 (A), +0,13 (B) e +0,67 (C). O paciente B fica perto do real ao longo de quase toda a trajetória.
- A imagem final desta execução não é o Dsr entregue (gerador aleatório diferente), mas o deslocamento é do mesmo tamanho (+0,58 aqui; +0,61 no Dsr entregue).

Figuras: `figuras/trajetoria_nex_por_paciente.png` (mediana por paciente; Real e A(R) à esquerda) e `figuras/trajetoria_x0_por_corte.png` (uma linha por corte, NEx e GLCM).

## 3. NEx e GLCM medem coisas diferentes

Spearman por corte (`dados/correlacoes_nex_glcm.csv`):

- **Dentro de cada classe**, NEx quase não se correlaciona com a GLCM: no real, |ρ| ≤ 0,05 com contraste e energia.
- **Nas mudanças sintético − real**, Δ NEx não acompanha Δ contraste (ρ = 0,05) nem Δ energia (ρ = 0,10). A redundância está entre as duas GLCM (Δ contraste × Δ energia: ρ = −0,66).
- Real e sintético agrupados dão NEx × contraste ρ = −0,69, mas isso é só a separação entre as classes (paradoxo de Simpson), não covariação.

Para o framework: o NEx acrescenta informação que contraste e energia GLCM não dão.

## Leitura

- A assinatura nasce na difusão/amostragem, não na passagem isolada pelo AEKL. Um motivo de revisão baseado no NEx deve descrever o que se mede ("NEx acima da faixa real"), **sem atribuir a causa ao decodificador**.
- Isto localiza a assinatura; **não** mostra que ela prejudica o uso das imagens. Se ela atrapalha o data augmentation é a próxima pergunta, ainda aberta.

## Limitações

- 3 pacientes, com 78 dos 100 cortes num só. Nada aqui generaliza para além deles.
- A(R) usa `z_mu`; o efeito de amostrar o latente não foi reproduzido. Não houve ablação de condicionamento, scheduler ou número de passos: "difusão/amostragem" é tudo o que vem depois do AEKL isolado.
- O comando original que gerou o Dsr entregue não foi preservado. O runner usado em 02/10 reconstrói o caminho de inferência a partir do código de treino e do W&B (`run-20250728_200033-u8moqqlw`); a configuração efetiva ainda está pendente de confirmação com quem treinou o modelo.
- A análise de 02/10 é exploratória e pós-hoc; a de 25/09 tinha leitura pré-especificada.
- Região: a imagem inteira com fundo harmonizado mistura corpo e pulmão. Na máscara do Dsr, 0,5 = tecido mole e 1 = parênquima; o MVP usa só o parênquima.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `dados/etapa_b_resultado.json`, `dados/etapa_b_config.json` | números e configuração da reconstrução pelo AEKL |
| `dados/reconstrucao_aekl_escores.csv` | NEx e GLCM de A(R), por corte |
| `dados/trajetoria_decis_escores.csv` | NEx e GLCM por corte em cada decil (real, x_t, x̂_0) |
| `dados/trajetoria_decis_config.json` | configuração e hashes da trajetória |
| `dados/correlacoes_nex_glcm.csv` | correlações NEx × GLCM por grupo |

Os CSVs só têm índice opaco de corte e apelido de paciente (A, B, C), sem nomes de arquivo.

## Proveniência

- **Reconstrução pelo AEKL (25/09):** `nyquist-excess`, commit `46f87be` (execução `20260925T0716`; dois arquivos ainda fora do commit na hora da execução, listados em `etapa_b_config.json`), scripts `12_aekl_reconstruct.py` e `13_etapa_b_scores.py`, protocolo `docs/etapa-b/PROTOCOLO.md`. Checkpoint AEKL `model_aekl_best_4_latent_new_lung.pth`, SHA-256 `f9cf5ec7…`.
- **Trajetória (02/10):** execução `20261002T1237`, scripts `15_decis_trajetoria.py` (SHA-256 `032476fe…` na execução), `16_decis_figuras.py` e `17_correlacoes.py`; síntese no ARIA com `synthesize_dsr_deciles.py` (SHA-256 `9de1a629…`). Checkpoint LDM `model_ldm_pulmao_Dsr.pth`, SHA-256 `0fadae69…`. **Esses scripts ainda não estão commitados** no `nyquist-excess` (branch `p0-provenance`, HEAD `f20714a`); quando forem, este parágrafo deve ganhar o hash do commit.
