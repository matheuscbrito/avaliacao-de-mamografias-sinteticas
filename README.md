# Avaliação de CT pulmonar sintética

Framework para responder, corte a corte, se uma CT pulmonar sintética é parecida o bastante com a CT real que a originou. Para cada par **real × sintética** ele dá duas respostas independentes:

- **O brilho está certo?** → aprovada, revisão manual ou rejeição direta.
- **A textura está certa?** → aprovada, revisão ou reprovada.

Em cada resposta o painel mostra **qual técnica decidiu e por quê** (ex.: "GLCM reprovou: energia passou de ±2 IQR").

O framework faz uma triagem e não substitui a avaliação radiológica: aprova o que está claramente dentro da variação entre CTs reais e manda o resto para revisão humana.

## Como funciona

### 1. Mesma região e mesma régua para todas as técnicas

- Todas as medidas são feitas só dentro dos pulmões (máscara erodida com raio 2), nos mesmos pares.
- GLCM e Wavelet usam o mesmo recorte de intensidade, calibrado só nos pixels das CTs reais (cercas de Tukey, k = 3). Com isso, as métricas de textura ficam cegas a mudanças de brilho puro, e brilho e textura não se contaminam.

### 2. A régua: IQR das CTs reais

Para cada métrica:

> Δ = (valor da sintética − valor da real) ÷ **IQR dessa métrica entre as CTs reais**

O IQR (intervalo interquartil) mede o quanto pulmões reais variam entre si. A pergunta que a régua responde é: *a sintética se afastou da original mais do que pulmões reais costumam diferir?*

| \|Δ\| | classificação |
|---|---|
| até 1 IQR | aprovada |
| de 1 a 2 IQR | revisão |
| acima de 2 IQR | reprovada |

O IQR vem só das CTs reais e fica congelado no histórico. Um gerador ruim não afrouxa a própria régua, e versões diferentes do gerador são comparadas com o mesmo critério. Dsr e Dgen usam as mesmas 37 CTs reais, então a régua dos dois é idêntica.

### 3. Filtro de brilho: GLCM + Wavelet LL

| técnica | o que mede | aprovada | revisão manual | rejeição direta |
|---|---|---|---|---|
| **GLCM · \|Δμ\|** | diferença do tom médio do pulmão (régua de 32 tons) | ≤ 0,314 | 0,314 a 0,629 | > 0,629 |
| **GLCM · Sat31** | % de pixels no tom mais claro (31) | ≤ 15% | — | > 15% |
| **Wavelet · LL** | média local de 4×4 px (baixa frequência: só o brilho) | ≤ 1 IQR | 1 a 2 IQR | > 2 IQR |

- O veredito de brilho é o da técnica mais severa.
- Os limites de \|Δμ\| são 1 e 2 IQR das médias pulmonares reais.
- A LL entra porque acompanha o brilho e pega cortes claros que o Δμ deixa passar. Energia e entropia da Wavelet não respondem ao brilho, por isso ficam só na textura.

### 4. Análise de textura: GLCM + Wavelet + NEx

| técnica | métricas | regra |
|---|---|---|
| **GLCM** | contraste (muda entre pixels vizinhos; cai quando a imagem suaviza) e energia (uniformidade; sobe quando fica lisa) | régua de IQR; a pior métrica decide |
| **Wavelet** | energia em 2 px (alta frequência, textura fina), em 4 px (média frequência, textura grossa) e entropia | régua de IQR; o "tipo" (borrou, ruído, perdeu contraste) descreve a mudança, não decide |
| **NEx** | excesso ou falta de estrutura perto do limite de resolução | fora da faixa P5–P95 das reais = revisão; nunca reprova |

O veredito de textura é o da técnica mais severa. O NEx não reprova por três motivos: a faixa P5–P95 deixa 10% das CTs reais de fora por definição, ele ainda não tem um segundo limite, e ainda não se sabe se a assinatura que ele detecta atrapalha o uso final das imagens.

## Resultados atuais (37 cortes por dataset)

| dataset | como a sintética foi gerada | textura (aprovada / revisão / reprovada) | brilho (aprovada / revisão / rejeição) |
|---|---|---|---|
| **Dsr** | a partir da própria CT em baixa resolução (mesma anatomia) | 8 / 29 / 0 | 35 / 2 / 0 |
| **Dgen** | só a partir do contorno pulmonar (outro pulmão no mesmo contorno) | 2 / 26 / 9 | 18 / 14 / 5 |

- O Dsr preserva o brilho, mas mexe na textura: quase tudo vai para revisão e nada é reprovado.
- O Dgen erra nas duas frentes, como esperado para um gerador que cria um pulmão novo.
- No Dgen, a Wavelet LL pegou 3 cortes claros demais que passavam pelo filtro da GLCM.

## Como rodar

Na raiz do repositório, com Python 3.11 ou mais novo:

```bash
python3 -m venv .venv
.venv/bin/pip install numpy pandas scipy scikit-image PyWavelets matplotlib Pillow streamlit altair
```

Os dados ficam em `dados/` (não versionado), com um par por corte: `X.tiff` (real), `X_generate.tiff` (sintética) e `X_mask.tiff` (máscara pulmonar). Hoje há `dados/Dsr` e `dados/Dgen`; as máscaras ficam em `dados/Dgen`.

### Painel (o jeito principal de usar)

```bash
.venv/bin/streamlit run frameworks/texture_evaluator/ui_placar.py --server.port 8515 --theme.base light --theme.font monospace
```

Abra http://localhost:8515. No primeiro acesso a cada dataset, o painel oferece avaliá-lo e cria o histórico em `outputs/`.

1. **Barra lateral:** escolha o dataset (Dsr ou Dgen) e a aba ("Análise de textura" ou "Filtro de brilho").
2. **Topo:** a tabela com os limites de aprovada, revisão e reprovada da aba.
3. **Resumo do dataset:** quantos cortes em cada categoria e quantos cada técnica apontou.
4. **Par em foco:** aparece quando você escolhe um corte. Mostra o veredito combinado, um cartão por técnica com o motivo e as duas CTs lado a lado.
5. **Todos os cortes:** matriz corte × técnica, gráficos por técnica e tabela. Clicar num ponto, célula ou linha abre aquele par.

Se o painel avisar que o histórico é antigo, clique em **"Recalcular versões"**.

### Linha de comando

Avaliação completa de um dataset, com tabela e relatório HTML em `outputs/mvp/report/index.html`:

```bash
.venv/bin/python -m frameworks.texture_evaluator.run_mvp \
  --data-dir dados/Dsr --mask-dir dados/Dgen --output-dir outputs/mvp
```

Placar com várias versões do gerador (a primeira versão vira a referência da régua):

```bash
.venv/bin/python -m frameworks.texture_evaluator.placar --output-dir outputs/placar \
  --version v1=CAMINHO/DA/VERSAO1 --mask-dir dados/Dgen
```

## Estrutura

| pasta | conteúdo |
|---|---|
| `frameworks/texture_evaluator/` | o avaliador: medição (`run_mvp.py`), regras de decisão (`decision.py`), histórico de versões (`placar.py`) e painel (`ui_placar.py`). Detalhes em [`README.md`](frameworks/texture_evaluator/README.md) |
| `frameworks/texture_evaluator/glcm/` | módulo GLCM (régua de 32 tons, contraste e energia). Ver [`README.md`](frameworks/texture_evaluator/glcm/README.md) |
| `wavelet/` | módulo Wavelet (energia, entropia, LL, tipo de defeito e mapa por região). Ver [`docs/wavelet.md`](docs/wavelet.md) |
| `docs/contracts/` | contratos de integração ([textura](docs/contracts/texture_evaluator.md), [brilho](docs/contracts/brightness_filter.md)) |
| `docs/protocols/` | decisões metodológicas oficiais |
| `resultados/` | uma pasta por análise, cada uma com `resumo.md` (ex.: [origem da assinatura NEx](resultados/origem_assinatura_nex/resumo.md), [régua única](resultados/regua_unica/resumo.md), [teste com defeitos conhecidos](resultados/teste_decisao/resumo.md)) |
| `archive/` | material histórico (código, figuras, slides) |
| `dados/`, `outputs/` | dados locais e saídas geradas; não versionados |

## Limitações e próximos passos

- **A régua é branda.** A textura varia muito entre pulmões reais, então diferenças grandes ainda podem aprovar. Há um corte com 60% do contraste original que aprova. A próxima régua a testar é 1,5 / 2,5 IQR, e o ideal é calibrar os limites com a avaliação visual de um radiologista.
- **Comparar com pulmões reais, não só com a original do par.** No Dgen, a "original" é outro pulmão no mesmo contorno, então as leituras ponto a ponto ficam como n/a. Medir se cada corte cabe na faixa das CTs reais serviria para qualquer gerador.
- **NEx:** decidir se ganha um nível de reprovação e verificar se a assinatura dele prejudica o data augmentation.
- **Relatório × painel:** o veredito salvo no relatório (`final_status`) ainda usa as regras antigas, e pode discordar do painel.
- **Os cortes não são independentes.** Há fases do mesmo paciente entre os 37 cortes.
- **AUC alta não é selo de qualidade.** ROC/AUC mostra o quanto reais e sintéticas são distinguíveis, não a qualidade clínica da síntese.
