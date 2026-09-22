# Framework de validação de textura — CT pulmonar sintética

## Objetivo

O framework responde a uma pergunta prática: **esta imagem de tomografia pulmonar sintética pode entrar no dataset de data augmentation?**

Para isso, ele avalia cada imagem sintética individualmente, sempre junto da imagem original que a condicionou. A análise busca três coisas:

1. preservar as características de textura do caso original;
2. continuar compatível com a variabilidade observada em pulmões reais validados;
3. não introduzir sinais típicos de geração artificial, como textura excessivamente lisa, padrões repetidos ou perda de detalhes finos.

Nesta primeira versão, o framework é calibrado especificamente para o dataset atual. Ele não deve ser tratado, ainda, como uma regra universal para qualquer CT ou protocolo de aquisição.

## Região analisada

A textura é medida apenas dentro do pulmão:

1. a máscara pulmonar da imagem original define a região de interesse;
2. a máscara passa por uma pequena erosão na borda, removendo a transição intensa entre pulmão e tecido externo;
3. a mesma posição espacial da máscara é aplicada à imagem sintética pareada.

Isso é adequado porque a imagem sintética foi gerada a partir da original: as duas imagens estão alinhadas. Assim, comparamos o mesmo território pulmonar nas duas, sem deixar o fundo preto ou a borda do pulmão contaminarem as métricas.

## Entradas e referência interna

| Papel | Conteúdo |
| --- | --- |
| Entrada do usuário | Uma imagem sintética e a imagem original que a condicionou. |
| Referência interna | Perfil estatístico gerado previamente a partir de imagens originais validadas. |
| Resultado | Métricas, comparação com o par, comparação com a referência e uma decisão de uso. |

O perfil de referência não precisa ser enviado a cada execução. Ele fica em segundo plano, versionado junto do framework, e contém a distribuição das métricas nas imagens reais aceitas, além dos parâmetros de pré-processamento usados para calculá-las.

No piloto atual, a referência foi preparada com 37 imagens originais centrais, contendo os dois pulmões e máscaras revisadas operacionalmente. Essa referência poderá crescer ou ser recalibrada quando houver mais casos validados.

## Pré-processamento fixo

Para que uma diferença reflita a imagem — e não uma mudança na forma de calcular — todas as imagens usam o mesmo procedimento:

- recorte de intensidade (*clipping/windowing*) definido a partir das originais de referência;
- discretização comum dos níveis de cinza;
- máscara erodida;
- mesmas direções e mesma distância entre pixels;
- cálculo apenas para pares de pixels que permanecem dentro da máscara.

Essas escolhas serão registradas no perfil de referência. Uma métrica calculada com parâmetros diferentes não deve ser comparada diretamente com a distribuição anterior.

## Métricas de textura selecionadas

O conjunto inicial tem quatro famílias complementares. Elas não tentam medir exatamente a mesma coisa.

### 1. GLCM — relação entre pixels vizinhos

A GLCM observa como intensidades de pixels próximos aparecem juntas. Ela descreve a microtextura local do pulmão.

- **Contraste:** mede quanto os tons variam entre vizinhos. Contraste muito menor no sintético pode indicar *over-smoothing*: uma textura lisa demais, que perdeu a heterogeneidade natural do parênquima. Contraste muito maior pode sugerir ruído ou artefato excessivo.
- **Energia:** mede o quanto alguns padrões locais se repetem de forma dominante. Energia muito alta pode sinalizar uma textura “estampada”, regular ou artificialmente repetitiva; energia muito baixa pode indicar perda de organização local.

Nesta etapa, contraste e energia são as duas features GLCM principais. Correlação e homogeneidade podem ser investigadas depois, mas não entram como decisoras iniciais por terem interpretação menos direta ou sobreposição parcial com contraste.

### 2. GLRLM — continuidade de padrões

A GLRLM mede sequências de pixels vizinhos com intensidade parecida, chamadas de *runs*. Ela acrescenta uma visão que a GLCM não fornece tão diretamente: por quanto tempo uma textura permanece contínua.

As candidatas iniciais são:

- **ênfase em runs curtos:** ajuda a detectar predominância de detalhes finos e variações pequenas;
- **ênfase em runs longos:** indica áreas grandes e uniformes; valor elevado demais pode reforçar o sinal de alisamento excessivo.

A seleção final dessas features dependerá dos testes de estabilidade e redundância descritos nos próximos passos.

### 3. Power Spectrum — distribuição por escala

O espectro de potência mostra como a energia visual está distribuída entre estruturas largas e detalhes finos.

- **Inclinação espectral:** resume se a imagem concentra mais energia em estruturas de grande escala ou em detalhes de alta frequência.
- **Fração de energia em alta frequência:** verifica a presença relativa de detalhes finos. Queda excessiva no sintético sugere perda de textura; aumento excessivo pode representar granulação ou ruído artificial.

Ele complementa a GLCM porque olha para a textura pelo ponto de vista da frequência, e não só das relações entre pixels vizinhos.

### 4. Wavelet — textura localizada em várias escalas

Wavelet separa a imagem em componentes de baixa e alta frequência em diferentes escalas e posições. Diferentemente do Power Spectrum, ela preserva também a localização do detalhe.

As candidatas iniciais são:

- **proporção de energia nos sub-bands de alta frequência:** mede quanto detalhe local foi preservado;
- **entropia dos sub-bands:** mede diversidade e complexidade dos padrões de detalhe.

Wavelet é uma transformação, não uma métrica única. As features finais serão escolhidas após a avaliação, para evitar acumular medidas redundantes.

## Como a decisão é feita

Cada feature é lida em dois eixos complementares.

### Eixo A — fidelidade ao caso que condicionou a geração

Para cada métrica, calculamos a diferença entre a sintética e sua original pareada:

`diferença do par = feature(sintética) − feature(original)`

Essa pergunta é: **a geração preservou a textura específica daquele pulmão?**

### Eixo B — compatibilidade com pulmões reais validados

A mesma feature da sintética é posicionada na distribuição da referência interna: mediana, dispersão, percentil e faixas de alerta.

Essa pergunta é: **mesmo preservando o par, a imagem continua plausível dentro da população real deste dataset?**

Não buscamos um valor idêntico à média ou à mediana. Pulmões reais variam. Uma imagem pode ser rara e ainda ser válida; por isso, a posição da original na própria distribuição também é usada para interpretar a posição da sintética.

## Casos possíveis e retorno

| Comparação com o par | Referência de originais validadas | Interpretação | Retorno do framework | Data augmentation |
| --- | --- | --- | --- | --- |
| Próxima | Dentro da faixa esperada | Preservou o caso e permanece realista. | **Aprovada** | Entra. |
| Próxima | Sintética e original são raras de modo compatível | Caso incomum, mas a geração preservou uma característica real do par. | **Aprovada com observação** | Entra, com etiqueta de caso raro. |
| Distante | Ainda dentro da faixa esperada | Pode ter havido regressão à média ou mudança de textura relevante, mesmo sem sair da população. | **Revisão manual** | Fica de fora até revisão. |
| Distante | Fora da faixa esperada | Forte indício de falha de geração ou artefato. | **Rejeitada** | Não entra. |

Uma original só pode servir como condicionante se tiver sido previamente validada. Uma original ruidosa ou artefatual não deve legitimar uma sintética semelhante.

## Saída esperada para uma imagem

O relatório por imagem deverá trazer:

- valor de cada feature na original e na sintética;
- diferença do par e sua interpretação;
- percentil da original e da sintética na referência real;
- sinalização por feature: normal, atenção ou fora da faixa;
- decisão final: **aprovada**, **aprovada com observação**, **revisão manual** ou **rejeitada**;
- motivo legível da decisão, por exemplo: “contraste menor que o par e abaixo da referência, sugerindo alisamento excessivo”.

A decisão final não deve depender cegamente de uma única métrica. O relatório precisa mostrar quais sinais sustentaram o resultado para permitir auditoria e revisão humana.

## Próximos passos: avaliação das features

Antes de tornar uma feature parte fixa do framework, ela precisa passar por quatro testes.

1. **Complementaridade:** medir algo que as features já escolhidas não capturam bem.
2. **Estabilidade:** manter comportamento consistente diante de pequenas mudanças razoáveis de máscara, erosão, discretização e janela de intensidade.
3. **Baixa redundância:** não ser quase uma cópia de outra feature. Nas originais validadas, verificaremos a correlação entre features; correlação muito alta é sinal de sobreposição.
4. **Interpretação clara:** uma alteração precisa ser traduzível em hipótese visual útil, como alisamento, repetição, perda de detalhe, continuidade excessiva ou ruído.

Também será avaliado se a feature varia o suficiente entre imagens reais válidas e se consegue distinguir transformações controladas — por exemplo, uma versão propositalmente suavizada ou ruidosa — sem reagir de modo aleatório.

## Ordem de implementação

1. Consolidar a referência de GLCM nas originais validadas: contraste e energia.
2. Definir como normalizar a diferença entre sintética e original para cada feature.
3. Implementar o relatório por par para GLCM, sem ainda tomar uma decisão automática definitiva.
4. Avaliar as features candidatas de GLRLM, Power Spectrum e Wavelet pelos quatro critérios acima.
5. Incorporar somente as features aprovadas e definir a regra integrada de aprovação, revisão e rejeição.
6. Gerar exemplos visuais de cada caso para a documentação e os slides.

## Princípio de evolução

O framework começa pequeno e explicável. A prioridade não é extrair o maior número possível de métricas, e sim manter apenas as que contribuam com uma evidência diferente, estável e interpretável para decidir se uma sintética agrega valor ao dataset.
