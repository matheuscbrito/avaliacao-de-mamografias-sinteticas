# Resumo para slides — controle de qualidade de imagens sintéticas

## Slide 1 — Objetivo

- Avaliar se imagens sintéticas são adequadas para treinar modelos médicos.
- Comparar imagens reais e sintéticas dentro das mesmas regiões de interesse.
- Separar diferenças de tecido, artefatos do gerador e erros de representação.
- Relacionar o controle de qualidade ao desempenho do modelo downstream.

## Slide 2 — EDA inicial

- Analisamos 225 pares de ROIs de mamografia.
- O conjunto representa 202 pacientes.
- Foram extraídas e comparadas 173 métricas.
- Usamos controles com blur, interpolação e ruído.
- Muitas métricas separavam os grupos por motivos triviais.

## Slide 3 — Problemas encontrados nos dados

- Real e sintético estavam em faixas de intensidade diferentes.
- O real tinha quantização de 12 bits; o sintético era float contínuo.
- Sem correção, algumas métricas mediam escala ou quantização, não textura.
- Normalização, quantização e pré-processamento precisam ser iguais.

## Slide 4 — Resultado das 173 métricas

- 82 métricas não detectaram diferenças úteis.
- 42 indicaram diferença em escala de tecido.
- 19 indicaram diferença apenas em escala fina.
- 18 foram inconclusivas.
- 8 não encontraram diferença relevante naquele eixo.
- 4 atuaram como detectores de artefatos de geração.

## Slide 5 — Power Spectrum e GLCM

- O Power Spectrum mostrou redistribuição de energia entre escalas.
- Houve redução de potência perto de 32 pixels.
- Houve aumento de potência abaixo de 16 pixels.
- Em 4 pixels, a razão sintético/real chegou a aproximadamente 2,5.
- GLCM também indicou mudanças na textura local.
- Power Spectrum será central; GLCM será complementar.

## Slide 6 — O achado próximo a Nyquist

- Um escore apresentou forte separação entre real e sintético.
- Mediana real: 0,61.
- Mediana sintética: 2,67.
- O sintético teve valor maior nos 225 pares.
- A assinatura apareceu nos três lotes de geração.
- O resultado ainda é exploratório.

## Slide 7 — O que esse escore significa

- O código compara um pico de alta frequência com o fundo espectral.
- O resultado é expresso como `log10(pico/fundo)`.
- Valor 1 representa uma razão de 10 vezes.
- Valor 2 representa uma razão de 100 vezes.
- Isso não mede profundidade de bits nem contraste direto entre pixels.
- A implementação ainda precisa ser validada.

## Slide 8 — Por que precisamos validar

- A métrica surgiu depois de observar os resultados.
- Ela não foi definida antecipadamente.
- O código foi desenvolvido com auxílio de LLMs.
- O último bin analisado não coincide exatamente com Nyquist.
- Blur, quantização, bordas e máscaras podem alterar o valor.
- Separar real de sintético não prova relevância clínica.

## Slide 9 — O que a literatura sustenta

- Geradores podem produzir picos periódicos no espectro.
- Upsampling pode criar checkerboard e aliasing.
- Difusão latente também pode deixar assinaturas espectrais.
- A frequência do padrão pode acompanhar o fator de upsampling.
- Não existe uma métrica clínica padronizada chamada `nyquist_excess`.
- A literatura justifica investigar o fenômeno, não valida nosso código.

## Slide 10 — Dados pulmonares

- Temos 100 pares de CT real e sintética.
- O conjunto representa apenas três pacientes.
- Os TIFFs não preservam metadados DICOM nem unidades HU.
- O sintético apresentou menos ruído e menor nitidez nas métricas iniciais.
- Esse conjunto será usado como bancada metodológica.

## Slide 11 — Gerador recuperado

- Recuperamos o peso e a configuração do AEKL.
- Identificamos a run original do DGen.
- O DGen trabalha no espaço latente e usa máscara como condicionamento.
- O treinamento original executou 10 épocas.
- O código atual contém alterações posteriores à geração analisada.

## Slide 12 — Ablação 1: validar a métrica

- Criar padrões artificiais com frequência conhecida.
- Testar alternância horizontal, vertical e checkerboard.
- Testar ruído, blur, quantização e diferentes tamanhos de ROI.
- Comparar o estimador atual com FFT 2D.
- Comparar com autocorrelação e contraste par–ímpar.
- Manter apenas medidas específicas, estáveis e interpretáveis.

## Slide 13 — Ablação 2: localizar a origem

- Grupo 1: imagem real.
- Grupo 2: real reconstruída pelo AEKL.
- Grupo 3: imagem gerada pelo DGen.
- Real diferente do AEKL sugere participação do autoencoder.
- AEKL próximo do real e DGen diferente sugere difusão ou amostragem.
- Um resultado intermediário sugere que a difusão amplifica o artefato.

## Slide 14 — Ablação 3: relevância downstream

- Criar versões com atenuação leve, média e forte.
- Usar também um processamento placebo.
- Medir se um classificador fixo muda suas decisões.
- Treinar modelos com real + cada versão sintética.
- Manter quantidade de dados, classes e hiperparâmetros iguais.
- Avaliar somente em pacientes reais independentes.

## Slide 15 — Como interpretar o resultado

- Menos assinatura + melhor desempenho: possível efeito prejudicial.
- Menos assinatura + desempenho igual: artefato técnico sem efeito observado.
- Menos assinatura + pior desempenho: filtragem pode remover informação útil.
- Mudança sem relação gradual: outros atributos provavelmente interferiram.
- Remover o pico não garante preservação anatômica ou clínica.

## Slide 16 — Próxima entrega

- Auditoria do estimador em sinais controlados.
- Comparação real × AEKL × DGen.
- Gráficos espectrais e de autocorrelação.
- Painel pequeno de GLCM.
- Conclusão provisória sobre a origem do padrão.
- Protocolo fechado para o teste no classificador.

## Mensagem final

- Encontramos uma assinatura forte, mas ainda não validada.
- Primeiro confirmaremos o que a métrica realmente mede.
- Depois localizaremos em qual etapa o padrão aparece.
- Por fim, testaremos se sua atenuação melhora o uso dos dados sintéticos.
- O objetivo não é apenas distinguir imagens, mas prever sua utilidade.

## Figuras prontas

- `plots/nyquist_excess_distribuicoes.png`
- `plots/nyquist_excess_diferenca_pareada.png`
- `plots/nyquist_excess_por_lote.png`
