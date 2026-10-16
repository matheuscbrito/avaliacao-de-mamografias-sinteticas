# Contrato de integração do avaliador de textura

Especificação da junção de GLCM, Wavelet e NEx, implementada no MVP (`frameworks/texture_evaluator/run_mvp.py`).

## Entrada comum

Cada execução deve receber um manifesto com `case_id`, caminho da imagem real, caminho da imagem sintética, máscara pulmonar e versão do protocolo. A ROI deve ser erodida uma única vez, com raio 2, antes de ser entregue a qualquer métrica.

## Saída comum

Cada módulo deve devolver uma linha por imagem com `case_id`, `image_type` (`real` ou `synthetic`), nome e versão da métrica, parâmetros e seus valores. A comparação pareada só ocorre depois que as três saídas compartilham o mesmo conjunto de `case_id`.

## Regras

- Não misturar resultados de versões diferentes de pré-processamento.
- GLCM usa a régua robusta calibrada nos reais; Wavelet e NEx devem declarar explicitamente qualquer normalização própria.
- NEx não tem normalização própria: é invariante a `a·I + b` (brilho e ganho). Na ROI, o NEx é a mediana de janelas de 32 px inteiramente dentro da ROI erodida; a faixa real nessa ROI (mediana 0,03, IQR 0,23, P95 0,25) está em [`resultados/nex_referencia_roi/resumo.md`](../../resultados/nex_referencia_roi/resumo.md). Os números de `origem_assinatura_nex` são da imagem inteira e não servem de limite no MVP.
- NEx não identifica a causa da diferença. No Dsr, a assinatura aparece na difusão/amostragem, não no AEKL isolado ([`resultados/origem_assinatura_nex/resumo.md`](../../resultados/origem_assinatura_nex/resumo.md)); motivos de revisão baseados nele não devem atribuí-la ao decodificador.
- ROC/AUC avalia capacidade de detectar diferenças entre reais e sintéticas. AUC alta é alerta de separação, não selo de qualidade da síntese.

## Classificação por métrica

- GLCM (contraste, energia), Wavelet (2 px, 4 px, entropia) e Wavelet LL (brilho) são classificadas pela diferença sintética − original em IQR da métrica entre as originais de referência: |Δ| ≤ 1 `aprovada`, 1–2 `revisão`, > 2 `reprovada`. Os IQRs (`glcm_iqr`, `wavelet_iqr`, `ll_iqr`) ficam na calibração do histórico.
- NEx usa a própria regra: fora de P5–P95 das originais = `revisão`; poucas janelas = `inconclusivo`, que não pesa.
- Colunas por corte: `{glcm,wavelet}_<métrica>_{real,synthetic,z,status}`, `glcm_status`, `wavelet_status` (a pior métrica de cada técnica), `wavelet_ll_z`, `wavelet_ll_status` e `nex_status`.
- Veredito de textura = pior classificação entre GLCM, Wavelet e NEx (`decision.texture_status`). Ele não altera `final_status`, em que só o filtro de brilho rejeita.
