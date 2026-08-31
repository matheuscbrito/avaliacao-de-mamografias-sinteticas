"""
glcm_probe.py

Script de teste (probe) para a Camada de Validade de Textura do framework:
carrega imagens .npz de mamografia, faz um crop de ROI simples (evitando
fundo preto e marcadores), roda o GLCM em 4 angulações, e imprime tudo de
forma explicada: qual imagem, qual ROI, quais parâmetros, e o que cada
feature está dizendo.

Uso:
    python glcm_probe.py

Ajuste PASTA_IMAGENS e N_IMAGENS abaixo antes de rodar.
"""

import os
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt

# A extração de ROI (silhueta da mama, evitar fundo/marcador/peitoral, escolha
# do recorte) vive agora num módulo próprio, reutilizável por qualquer métrica.
from roi_extraction import (
    load_image as _re_load,
    extract_roi as _re_extract,
    _robust_background_threshold as _re_bg_threshold,
)

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO — ajuste aqui
# ---------------------------------------------------------------------------
PASTA_IMAGENS = "2d_only_1000_images/images"   # caminho da pasta com os .npz
N_IMAGENS = 3                                   # quantas imagens testar (só usado se ARQUIVOS_ESPECIFICOS estiver vazio)
TAMANHO_ROI = 128                               # crop quadrado em pixels (lado)
NIVEIS_CINZA = 32                               # quantização para o GLCM (8/16/32/64)
DISTANCIA = 1                                   # distância entre pixel e vizinho (em pixels)
ANGULOS = {
    "0°":   0,
    "45°":  np.pi / 4,
    "90°":  np.pi / 2,
    "135°": 3 * np.pi / 4,
}

# Preencha com nomes exatos de arquivos .npz para analisar SÓ esses,
# na ordem que você colocar (útil para revisar outliers específicos do
# analisar_baseline_csv.py, ou qualquer imagem que você queira conferir
# manualmente). Deixe a lista vazia [] para usar o comportamento padrão
# (as primeiras N_IMAGENS da pasta, em ordem alfabética).
ARQUIVOS_ESPECIFICOS = [
    # "100793095124884174010228504854086215788.npz",
    # "101038413610535901267185902654058053919.npz",
]

# Para o modo de comparação: liste aqui pares de arquivos que você julga
# visualmente parecidos (ou visualmente diferentes), para conferir se o
# GLCM concorda com o que o olho vê. Deixe a lista vazia [] para pular
# essa etapa. Preencha com os nomes exatos dos arquivos .npz da pasta.
COMPARAR_PARES = [
    # ("arquivo_A.npz", "arquivo_B.npz"),
]

# ---------------------------------------------------------------------------
# FUNÇÕES
# ---------------------------------------------------------------------------

def carregar_imagem(caminho):
    """Carrega a imagem (delegado a roi_extraction.load_image: .npz/.npy/.png/.dcm)."""
    return _re_load(caminho)


def calcular_limiar_tecido(img):
    """Limiar fundo/tecido — delegado a roi_extraction (mantido por compatibilidade)."""
    return _re_bg_threshold(np.asarray(img, dtype=np.float64))


def encontrar_roi_valida(img, tamanho, fracao_minima=0.7, passos_busca=15):
    """
    Wrapper de compatibilidade em volta de `roi_extraction.extract_roi`.

    A lógica de extração (silhueta da mama por limiar robusto + preenchimento a
    partir da parede torácica, checagens de sanidade que rejeitam espécime/
    plate/máscara fragmentada, região elegível erodida evitando pele/marcadores/
    peitoral, ROI no centróide do parênquima) agora vive em `roi_extraction.py`,
    para ser reaproveitada por qualquer métrica — não só o GLCM.

    Mantém a assinatura antiga: retorna (roi_array, (x0, y0, x1, y1), fracao).
    `passos_busca` é ignorado (não há mais busca em grade no caminho principal).
    Levanta ValueError se a imagem for rejeitada ou nenhuma ROI servir.
    """
    res = _re_extract(np.asarray(img, dtype=np.float64), size=tamanho,
                      min_tissue_fraction=fracao_minima)
    if res.rejected or not res.ok or res.roi is None:
        raise ValueError(res.reject_reason or "não foi possível extrair ROI válida")
    r = res.roi
    return r.array, r.bbox, r.tissue_fraction


def quantizar(roi, niveis):
    """Reduz a imagem para N níveis de cinza, exigido pelo GLCM."""
    roi_norm = (roi - roi.min()) / (roi.max() - roi.min() + 1e-8)
    roi_quant = (roi_norm * (niveis - 1)).astype(np.uint8)
    return roi_quant


def rodar_glcm(roi_quant, niveis):
    """
    Calcula o GLCM nas 4 angulações e extrai as 4 features clássicas:
    contraste, homogeneidade, energia e correlação.

    Retorna um dicionário {angulo: {feature: valor}} e a média por feature.
    """
    resultados = {}
    for nome_ang, ang_rad in ANGULOS.items():
        glcm = graycomatrix(
            roi_quant,
            distances=[DISTANCIA],
            angles=[ang_rad],
            levels=niveis,
            symmetric=True,
            normed=True,
        )
        resultados[nome_ang] = {
            "contraste": graycoprops(glcm, "contrast")[0, 0],
            "homogeneidade": graycoprops(glcm, "homogeneity")[0, 0],
            "energia": graycoprops(glcm, "energy")[0, 0],
            "correlacao": graycoprops(glcm, "correlation")[0, 0],
        }

    medias = {}
    for feat in ["contraste", "homogeneidade", "energia", "correlacao"]:
        medias[feat] = np.mean([resultados[ang][feat] for ang in ANGULOS])

    return resultados, medias


def explicar_feature(nome, valor):
    """Retorna uma frase curta interpretando o valor de cada feature."""
    if nome == "contraste":
        return "alto = textura mais grosseira/heterogênea; baixo = mais suave/uniforme"
    if nome == "homogeneidade":
        return "alto = pixels vizinhos parecidos entre si; baixo = mais variação local"
    if nome == "energia":
        return "alto = poucos padrões dominantes se repetindo; baixo = textura mais complexa"
    if nome == "correlacao":
        return "alto = forte dependência linear entre pixels vizinhos; baixo = mais aleatório"
    return ""


def interpretar_textura(medias, niveis):
    """
    Gera uma frase-resumo em linguagem natural interpretando o conjunto de
    features desta imagem, usando faixas de referência relativas ao número
    de níveis de cinza usado na quantização (para o contraste) e às escalas
    fixas 0-1 das demais features (homogeneidade, energia, correlação).

    Isso não substitui comparação entre imagens (que é o mais confiável),
    mas dá uma leitura qualitativa rápida de cada imagem isolada.
    """
    contraste = medias["contraste"]
    homogeneidade = medias["homogeneidade"]
    energia = medias["energia"]
    correlacao = medias["correlacao"]

    contraste_max_teorico = (niveis - 1) ** 2
    contraste_rel = contraste / contraste_max_teorico

    if contraste_rel < 0.02:
        desc_contraste = "textura bastante suave/uniforme"
    elif contraste_rel < 0.08:
        desc_contraste = "textura com heterogeneidade moderada"
    else:
        desc_contraste = "textura grosseira/heterogênea, com transições fortes entre regiões vizinhas"

    if energia > 0.15:
        desc_energia = "com poucos padrões dominantes se repetindo (mais previsível/repetitiva)"
    elif energia > 0.05:
        desc_energia = "com uma mistura moderada de padrões"
    else:
        desc_energia = "com muitos padrões diferentes coexistindo (textura complexa)"

    if correlacao > 0.85:
        desc_correlacao = "os pixels vizinhos são fortemente dependentes entre si (organização espacial clara)"
    elif correlacao > 0.6:
        desc_correlacao = "há uma dependência espacial moderada entre pixels vizinhos"
    else:
        desc_correlacao = "os pixels vizinhos variam de forma quase independente (aspecto mais ruidoso/aleatório)"

    frase = (
        f"Esta imagem apresenta {desc_contraste} "
        f"(contraste={contraste:.2f}, {contraste_rel*100:.1f}% do máximo teórico para {niveis} níveis), "
        f"{desc_energia} (energia={energia:.4f}), "
        f"e {desc_correlacao} (correlação={correlacao:.4f}). "
        f"A homogeneidade ficou em {homogeneidade:.4f} "
        f"({'alta, reforçando o aspecto suave' if homogeneidade > 0.5 else 'baixa, reforçando o aspecto heterogêneo'})."
    )
    return frase


def comparar_imagens(nome1, nome2, medias1, medias2, niveis):
    """
    Compara as features médias de duas imagens, calculando a diferença
    relativa em cada uma e uma distância geral normalizada. Serve como
    teste de sanidade: se duas imagens parecem visualmente parecidas,
    espera-se diferença pequena aqui; se parecem bem diferentes a olho
    nu, espera-se diferença maior.
    """
    contraste_max_teorico = (niveis - 1) ** 2
    feats_norm = {
        "contraste": (medias1["contraste"] / contraste_max_teorico,
                      medias2["contraste"] / contraste_max_teorico),
        "homogeneidade": (medias1["homogeneidade"], medias2["homogeneidade"]),
        "energia": (medias1["energia"], medias2["energia"]),
        "correlacao": (medias1["correlacao"], medias2["correlacao"]),
    }

    print(f"\nCOMPARANDO: {nome1}  vs  {nome2}")
    diffs = []
    for feat, (v1, v2) in feats_norm.items():
        diff_abs = abs(v1 - v2)
        diffs.append(diff_abs)
        print(f"  {feat:15s} -> {nome1[:20]:20s} = {v1:.4f}  |  "
              f"{nome2[:20]:20s} = {v2:.4f}  |  diferença = {diff_abs:.4f}")

    distancia = float(np.sqrt(np.sum(np.array(diffs) ** 2)))
    print(f"\n  Distância geral (euclidiana, features normalizadas 0-1): {distancia:.4f}")

    if distancia < 0.10:
        veredito = ("as métricas ficaram bem próximas — se essas duas imagens também "
                     "parecem com textura semelhante a olho nu, é um bom sinal de que o "
                     "pipeline está funcionando corretamente.")
    elif distancia < 0.25:
        veredito = ("as métricas ficaram moderadamente diferentes — vale conferir visualmente "
                     "se a diferença encontrada bate com o que se espera dessas duas imagens.")
    else:
        veredito = ("as métricas ficaram bem diferentes. Se a expectativa era de texturas "
                     "parecidas, vale investigar: pode ser diferença real de tecido, ou a ROI "
                     "pode ter caído em regiões não comparáveis (uma em tecido denso, outra em "
                     "gordura, por exemplo) — confira as marcações visuais das duas.")

    print(f"  Veredito: {veredito}")
    return distancia


def visualizar(img, roi_coords, roi_quant, nome_arquivo):
    """Mostra a imagem inteira com a ROI marcada, e a ROI já quantizada ao lado."""
    x0, y0, x1, y1 = roi_coords
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(img, cmap="gray")
    axs[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                    edgecolor="red", facecolor="none", linewidth=2))
    axs[0].set_title(f"Imagem completa\n{nome_arquivo}", fontsize=9)
    axs[0].axis("off")

    axs[1].imshow(roi_quant, cmap="gray")
    axs[1].set_title(f"ROI usada no GLCM\n({NIVEIS_CINZA} níveis de cinza)", fontsize=9)
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# EXECUÇÃO PRINCIPAL
# ---------------------------------------------------------------------------

def analisar_imagem(nome_arquivo, mostrar=True):
    """
    Executa o pipeline completo (carregar -> ROI -> quantizar -> GLCM) para
    um único arquivo, imprime os detalhes e retorna as médias das features.
    Reaproveitada tanto no loop principal quanto no modo de comparação.
    """
    caminho = os.path.join(PASTA_IMAGENS, nome_arquivo)
    if not os.path.exists(caminho):
        print(f"\nAVISO: arquivo não encontrado: '{caminho}' — pulando.")
        print("=" * 70)
        return None

    print(f"\nIMAGEM: {nome_arquivo}")
    img = carregar_imagem(caminho)
    print(f"  Formato original: {img.shape} | tipo: {img.dtype} | "
          f"min={img.min():.1f} max={img.max():.1f}")

    try:
        roi, roi_coords, frac_tecido = encontrar_roi_valida(img, TAMANHO_ROI)
    except ValueError as e:
        print(f"  AVISO: {e}")
        print("=" * 70)
        return None

    x0, y0, x1, y1 = roi_coords
    print(f"  ROI extraída: crop de {TAMANHO_ROI}x{TAMANHO_ROI}px, "
          f"posição (x={x0}:{x1}, y={y0}:{y1}), "
          f"fração de tecido na ROI = {frac_tecido:.0%}")

    roi_quant = quantizar(roi, NIVEIS_CINZA)
    print(f"  ROI quantizada para {NIVEIS_CINZA} níveis de cinza "
          f"(necessário para o cálculo do GLCM)")

    print(f"  Calculando GLCM: distância={DISTANCIA}px, "
          f"ângulos={list(ANGULOS.keys())}")

    resultados, medias = rodar_glcm(roi_quant, NIVEIS_CINZA)

    print("\n  Resultado por ângulo:")
    for ang, feats in resultados.items():
        linha = "    " + ang.rjust(5) + " -> "
        linha += " | ".join(f"{k}={v:.4f}" for k, v in feats.items())
        print(linha)

    print("\n  Média entre os 4 ângulos (valor-resumo por imagem):")
    for feat, valor in medias.items():
        print(f"    {feat:15s} = {valor:.4f}")

    print("\n  INTERPRETAÇÃO:")
    print(f"  {interpretar_textura(medias, NIVEIS_CINZA)}")

    if mostrar:
        print("\n  Exibindo imagem + ROI marcada...")
        visualizar(img, roi_coords, roi_quant, nome_arquivo)

    print("=" * 70)
    return medias


def main():
    if ARQUIVOS_ESPECIFICOS:
        arquivos_para_rodar = ARQUIVOS_ESPECIFICOS
        print(f"Modo arquivos específicos: analisando {len(arquivos_para_rodar)} "
              f"arquivo(s) escolhido(s) manualmente.\n")
    else:
        arquivos = sorted([f for f in os.listdir(PASTA_IMAGENS) if f.endswith(".npz")])
        print(f"Encontrei {len(arquivos)} arquivos .npz em '{PASTA_IMAGENS}'.")
        print(f"Vou analisar as primeiras {N_IMAGENS} agora.\n")
        arquivos_para_rodar = arquivos[:N_IMAGENS]
    print("=" * 70)

    resultados_por_imagem = {}
    for nome_arquivo in arquivos_para_rodar:
        medias = analisar_imagem(nome_arquivo)
        if medias is not None:
            resultados_por_imagem[nome_arquivo] = medias

    if COMPARAR_PARES:
        print("\n\n" + "#" * 70)
        print("# MODO COMPARAÇÃO — testando se imagens visualmente parecidas")
        print("# retornam métricas parecidas (teste de sanidade do pipeline)")
        print("#" * 70)
        for nome1, nome2 in COMPARAR_PARES:
            if nome1 not in resultados_por_imagem:
                resultados_por_imagem[nome1] = analisar_imagem(nome1, mostrar=False)
            if nome2 not in resultados_por_imagem:
                resultados_por_imagem[nome2] = analisar_imagem(nome2, mostrar=False)
            if resultados_por_imagem.get(nome1) is None or resultados_por_imagem.get(nome2) is None:
                print(f"\nPulando comparação {nome1} vs {nome2}: um dos arquivos não foi encontrado.")
                continue
            comparar_imagens(nome1, nome2,
                              resultados_por_imagem[nome1],
                              resultados_por_imagem[nome2],
                              NIVEIS_CINZA)

    print("\nFeito. Compare visualmente se as imagens com contraste/homogeneidade "
          "diferentes realmente parecem ter texturas diferentes a olho nu — "
          "esse é o teste de sanidade antes de fixar os parâmetros e escalar.")


if __name__ == "__main__":
    main()