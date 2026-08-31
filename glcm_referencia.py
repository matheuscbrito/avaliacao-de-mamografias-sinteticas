"""
glcm_referencia.py — parâmetro de referência de textura (GLCM) das imagens REAIS.

Roda a extração de ROI (via roi_extraction.py) + GLCM em TODAS as imagens de um
dataset, remove outliers estatísticos, e cospe a distribuição de referência de
**contraste** e **energia** — os números contra os quais as imagens sintéticas
serão comparadas depois (Camada 2 do framework de validação).

Unifica o que antes eram 3 scripts (probe + baseline + análise de outliers).
A extração de ROI continua num módulo separado (roi_extraction.py), porque é
compartilhada com as outras métricas (Power Spectrum/NPS, wavelets, LBP…).

--------------------------------------------------------------------------------
USO
--------------------------------------------------------------------------------
    # referência a partir de um dataset de mamografia real:
    python glcm_referencia.py 2d_only_1000_images/images

    # com PNGs de conferência (imagem + máscara da mama + ROI) para revisar:
    python glcm_referencia.py 2d_only_1000_images/images --previews

    # conferir 1 imagem (ROI extraída + GLCM por ângulo):
    python glcm_referencia.py caso.npz --inspect --previews caso.png

    # ajustes:
    python glcm_referencia.py DIR --pattern "*.npz" --roi-size 128 --levels 32 \
        --distance 1 --iqr-k 1.5 --json ref.json --previews previews/ --limit 50

Saídas (modo dataset):
  - tabela no terminal com mediana / IQR / média±dp de contraste e energia
  - JSON com a referência completa + valor por imagem + outliers + rejeitadas
  - (opcional) 1 PNG por imagem em --previews

Para comparar sintético vs real: rode o script nos DOIS datasets e compare as
distribuições dos JSONs (mediana/IQR, KS, Wasserstein).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from skimage.feature import graycomatrix, graycoprops

from roi_extraction import extract_roi, load_image, save_preview

ANGULOS_DEG = [0, 45, 90, 135]
ANGULOS_RAD = [np.deg2rad(a) for a in ANGULOS_DEG]


# ---------------------------------------------------------------------------
# GLCM
# ---------------------------------------------------------------------------
def quantizar(roi: np.ndarray, niveis: int) -> np.ndarray:
    """Reescala a ROI para [0, niveis-1] (exigido pelo GLCM)."""
    lo, hi = float(roi.min()), float(roi.max())
    norm = (roi - lo) / (hi - lo + 1e-8)
    return (norm * (niveis - 1)).astype(np.uint8)


def glcm_contraste_energia(roi: np.ndarray, niveis: int, distancia: int) -> dict:
    """
    GLCM nos 4 ângulos (0/45/90/135°) e a média por feature.
    Foco: contraste (heterogeneidade) e energia (repetição de padrão) — o par
    que o grupo adotou (homogeneidade ~ inverso do contraste; correlação
    mostrou-se pouco confiável sozinha).
    """
    q = quantizar(roi, niveis)
    glcm = graycomatrix(q, distances=[distancia], angles=ANGULOS_RAD,
                        levels=niveis, symmetric=True, normed=True)
    contraste = graycoprops(glcm, "contrast")[0]      # (4,)
    energia = graycoprops(glcm, "energy")[0]
    return {
        "contraste": float(np.mean(contraste)),
        "energia": float(np.mean(energia)),
        "contraste_por_angulo": {f"{a}deg": float(v) for a, v in zip(ANGULOS_DEG, contraste)},
        "energia_por_angulo": {f"{a}deg": float(v) for a, v in zip(ANGULOS_DEG, energia)},
        "anisotropia_contraste": float(np.std(contraste) / (np.mean(contraste) + 1e-9)),
    }


# ---------------------------------------------------------------------------
# ESTATÍSTICA
# ---------------------------------------------------------------------------
def resumo(valores: np.ndarray) -> dict:
    q1, q3 = np.percentile(valores, [25, 75])
    return {
        "n": int(valores.size),
        "media": float(valores.mean()),
        "dp": float(valores.std()),
        "mediana": float(np.median(valores)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(valores.min()),
        "max": float(valores.max()),
        "p5": float(np.percentile(valores, 5)),
        "p95": float(np.percentile(valores, 95)),
        "faixa_ref": [float(np.median(valores) - (q3 - q1) / 2),
                      float(np.median(valores) + (q3 - q1) / 2)],
    }


def marcar_outliers(registros: list[dict], iqr_k: float) -> tuple[list[dict], dict]:
    """Regra de Tukey em contraste E energia. Marca `outlier`/`motivo_outlier`."""
    c = np.array([r["contraste"] for r in registros])
    e = np.array([r["energia"] for r in registros])

    def limites(arr):
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        return q1 - iqr_k * iqr, q3 + iqr_k * iqr

    c_lo, c_hi = limites(c)
    e_lo, e_hi = limites(e)
    for r in registros:
        motivos = []
        if not (c_lo <= r["contraste"] <= c_hi):
            motivos.append("contraste")
        if not (e_lo <= r["energia"] <= e_hi):
            motivos.append("energia")
        r["outlier"] = bool(motivos)
        r["motivo_outlier"] = "+".join(motivos)
    return registros, {"contraste": [float(c_lo), float(c_hi)],
                       "energia": [float(e_lo), float(e_hi)]}


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------
def processar(dataset: str, pattern: str, roi_size: int, levels: int,
              distance: int, iqr_k: float, previews_dir: str | None,
              limit: int | None) -> dict:
    arquivos = sorted(glob.glob(os.path.join(dataset, pattern)))
    if limit:
        arquivos = arquivos[:limit]
    if not arquivos:
        sys.exit(f"nenhum arquivo em {dataset!r} com padrão {pattern!r}")

    print(f"Dataset: {dataset}")
    print(f"{len(arquivos)} imagens | ROI {roi_size}px | {levels} níveis | "
          f"distância {distance}px | ângulos {ANGULOS_DEG}\n")

    registros: list[dict] = []
    rejeitadas: list[dict] = []

    for i, path in enumerate(arquivos, 1):
        nome = os.path.basename(path)
        try:
            img = load_image(path)
            res = extract_roi(img, size=roi_size, source=nome)
        except Exception as ex:                       # arquivo corrompido etc.
            rejeitadas.append({"arquivo": nome, "motivo": f"erro ao ler/extrair: {ex}"})
            continue

        if previews_dir:
            try:
                save_preview(res, os.path.join(previews_dir, os.path.splitext(nome)[0] + ".png"))
            except Exception:
                pass

        if res.rejected or not res.ok or res.roi is None:
            rejeitadas.append({"arquivo": nome,
                               "motivo": res.reject_reason or "sem ROI válida"})
            continue

        g = glcm_contraste_energia(res.roi.array, levels, distance)
        registros.append({
            "arquivo": nome,
            "contraste": g["contraste"],
            "energia": g["energia"],
            "anisotropia_contraste": g["anisotropia_contraste"],
            "roi_bbox": list(res.roi.bbox),
            "roi_tissue_fraction": res.roi.tissue_fraction,
            "avisos": res.warnings,
        })

        if i % 50 == 0 or i == len(arquivos):
            print(f"  [{i}/{len(arquivos)}]  ok={len(registros)}  rejeitadas={len(rejeitadas)}")

    if not registros:
        sys.exit("nenhuma imagem processada com sucesso.")

    registros, limites_outlier = marcar_outliers(registros, iqr_k)
    limpos = [r for r in registros if not r["outlier"]]
    outliers = [r for r in registros if r["outlier"]]

    ref = {
        "contraste": resumo(np.array([r["contraste"] for r in limpos])),
        "energia": resumo(np.array([r["energia"] for r in limpos])),
    }

    return {
        "dataset": os.path.abspath(dataset),
        "n_total": len(arquivos),
        "n_processadas": len(registros),
        "n_rejeitadas": len(rejeitadas),
        "n_outliers": len(outliers),
        "n_referencia": len(limpos),
        "params": {"roi_size": roi_size, "levels": levels, "distance": distance,
                   "angles_deg": ANGULOS_DEG, "iqr_k": iqr_k},
        "limites_outlier_tukey": limites_outlier,
        "referencia": ref,
        "outliers": [{"arquivo": r["arquivo"], "contraste": r["contraste"],
                      "energia": r["energia"], "motivo": r["motivo_outlier"]}
                     for r in outliers],
        "rejeitadas": rejeitadas,
        "por_imagem": [{"arquivo": r["arquivo"], "contraste": r["contraste"],
                        "energia": r["energia"], "outlier": r["outlier"]}
                       for r in registros],
    }


def imprimir_tabela(out: dict) -> None:
    print("\n" + "=" * 70)
    print("REFERÊNCIA DE TEXTURA (GLCM) — IMAGENS REAIS")
    print("=" * 70)
    print(f"{out['n_total']} imagens | {out['n_rejeitadas']} rejeitadas (não-mamografia) "
          f"| {out['n_outliers']} outliers | {out['n_referencia']} na referência")
    for feat in ("contraste", "energia"):
        s = out["referencia"][feat]
        print(f"\n{feat.upper()}  (n={s['n']})")
        print(f"  mediana        = {s['mediana']:.4f}")
        print(f"  IQR (Q1–Q3)    = [{s['q1']:.4f}, {s['q3']:.4f}]")
        print(f"  média ± dp     = {s['media']:.4f} ± {s['dp']:.4f}")
        print(f"  p5 – p95       = [{s['p5']:.4f}, {s['p95']:.4f}]")
        print(f"  min – max      = [{s['min']:.4f}, {s['max']:.4f}]")
        print(f"  faixa de ref.  = [{s['faixa_ref'][0]:.4f}, {s['faixa_ref'][1]:.4f}]  "
              f"(mediana ± IQR/2)")
    print("\n" + "=" * 70)
    print("Uso: rode este mesmo script no conjunto SINTÉTICO e compare as "
          "distribuições (mediana/IQR, ou KS/Wasserstein). Contraste baixo demais "
          "= textura suavizada; energia alta demais = repetição de padrão.")
    print("=" * 70)


def inspecionar(path: str, roi_size: int, levels: int, distance: int,
                preview: str | None) -> None:
    """Modo 1-imagem: mostra a ROI extraída e o GLCM por ângulo, para conferir."""
    nome = os.path.basename(path)
    img = load_image(path)
    print(f"IMAGEM: {nome}  shape={img.shape}  min={img.min():.0f} max={img.max():.0f}")
    res = extract_roi(img, size=roi_size, source=nome)
    if preview:
        save_preview(res, preview if preview.endswith(".png") else os.path.join(preview, nome + ".png"))
        print(f"preview: {preview}")
    if res.rejected or not res.ok or res.roi is None:
        print(f"REJEITADA: {res.reject_reason or 'sem ROI'}")
        return
    r = res.roi
    print(f"ROI: bbox={r.bbox}  fração de tecido={r.tissue_fraction:.0%}"
          + (f"  avisos: {res.warnings}" if res.warnings else ""))
    g = glcm_contraste_energia(r.array, levels, distance)
    print(f"\n  {'ângulo':>7} | {'contraste':>10} | {'energia':>10}")
    for a in ANGULOS_DEG:
        print(f"  {str(a)+'°':>7} | {g['contraste_por_angulo'][f'{a}deg']:>10.4f} | "
              f"{g['energia_por_angulo'][f'{a}deg']:>10.4f}")
    print(f"  {'média':>7} | {g['contraste']:>10.4f} | {g['energia']:>10.4f}")
    print(f"\n  anisotropia do contraste (desvio/média entre ângulos) = "
          f"{g['anisotropia_contraste']:.1%}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="pasta com as imagens (.npz/.npy/.png/.dcm), "
                                    "ou um arquivo único com --inspect")
    ap.add_argument("--pattern", default="*.npz", help="glob dos arquivos (default *.npz)")
    ap.add_argument("--roi-size", type=int, default=128)
    ap.add_argument("--levels", type=int, default=32, help="níveis de cinza p/ o GLCM")
    ap.add_argument("--distance", type=int, default=1, help="distância do GLCM em pixels")
    ap.add_argument("--iqr-k", type=float, default=1.5, help="multiplicador de IQR p/ outlier (Tukey)")
    ap.add_argument("--json", default=None, help="caminho do JSON de saída")
    ap.add_argument("--previews", nargs="?", const="glcm_previews", default=None,
                    metavar="DIR", help="salva 1 PNG de conferência por imagem (default: glcm_previews/)")
    ap.add_argument("--limit", type=int, default=None, help="processa só as N primeiras (teste)")
    ap.add_argument("--inspect", action="store_true",
                    help="modo 1-imagem: trata `dataset` como um arquivo e mostra ROI + GLCM por ângulo")
    args = ap.parse_args(argv)

    if args.inspect:
        inspecionar(args.dataset, args.roi_size, args.levels, args.distance,
                    args.previews)
        return

    out = processar(args.dataset, args.pattern, args.roi_size, args.levels,
                    args.distance, args.iqr_k, args.previews, args.limit)
    imprimir_tabela(out)

    json_path = args.json or (os.path.basename(os.path.normpath(args.dataset)) + "_glcm_ref.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nJSON: {json_path}")
    if args.previews:
        print(f"previews: {args.previews}/")


if __name__ == "__main__":
    main()
