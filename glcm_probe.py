#!/usr/bin/env python3
"""
glcm_probe.py — teste rápido de GLCM numa imagem de mamografia.

Objetivo: jogar uma imagem (PNG/JPG/TIF/DICOM) e ver o que as features de
GLCM retornam, sem compromisso com o pipeline final (Camada 2 - Textura).

Uso:
    .venv/bin/python glcm_probe.py caminho/imagem.png
    .venv/bin/python glcm_probe.py caso.dcm --levels 64 --roi center --roi-frac 0.5
    .venv/bin/python glcm_probe.py img.png --json out.json --save-roi roi.png

Notas de método (para discussão no grupo):
- GLCM aqui = skimage.feature.graycomatrix / graycoprops.
- Quantização: a imagem é reescalada para `--levels` níveis antes da GLCM.
  Mamografia costuma ser 12-16 bit; usar 256 níveis deixa a matriz esparsa e
  instável. 32-64 é o intervalo usual na literatura de radiômica.
- Distâncias e ângulos default: d in {1,2,4} px, theta in {0,45,90,135}.
- `contrast/dissimilarity/homogeneity/ASM/energy/correlation` vêm do skimage.
  `entropy` é calculada à mão sobre a GLCM normalizada.
- Reporta média sobre os 4 ângulos (aprox. invariante a rotação) e o desvio
  entre ângulos (proxy de anisotropia — parênquima tem direção preferencial).
- GLRLM NÃO está aqui: skimage não implementa. Para GLRLM/GLSZM use PyRadiomics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from skimage.feature import graycomatrix, graycoprops

DISTANCES = [1, 2, 4]
ANGLES_DEG = [0, 45, 90, 135]
ANGLES_RAD = [np.deg2rad(a) for a in ANGLES_DEG]
SKIMAGE_PROPS = [
    "contrast",
    "dissimilarity",
    "homogeneity",
    "ASM",
    "energy",
    "correlation",
]


def load_image(path: Path) -> tuple[np.ndarray, dict]:
    """Retorna imagem 2D float + metadados. Suporta DICOM e formatos comuns."""
    suffix = path.suffix.lower()
    meta: dict = {"path": str(path), "loader": None}

    if suffix in {".dcm", ".dicom", ""} or _looks_like_dicom(path):
        import pydicom

        ds = pydicom.dcmread(str(path), force=True)
        arr = ds.pixel_array.astype(np.float64)
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
        arr = arr * slope + intercept
        photometric = str(getattr(ds, "PhotometricInterpretation", "")).strip()
        if photometric == "MONOCHROME1":
            # MONOCHROME1: valores altos = escuro. Inverte para o padrão visual.
            arr = arr.max() - arr
        meta.update(
            loader="pydicom",
            photometric=photometric or None,
            rescale_slope=slope,
            rescale_intercept=intercept,
            bits_stored=int(getattr(ds, "BitsStored", 0)) or None,
            shape=list(arr.shape),
        )
        if arr.ndim == 3:
            arr = arr.mean(axis=-1)
        return arr, meta

    from PIL import Image

    im = Image.open(path)
    if im.mode not in {"I", "I;16", "F", "L"}:
        im = im.convert("L")
    arr = np.asarray(im).astype(np.float64)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    meta.update(loader="pillow", pil_mode=im.mode, shape=list(arr.shape))
    return arr, meta


def _looks_like_dicom(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            fh.seek(128)
            return fh.read(4) == b"DICM"
    except OSError:
        return False


def extract_roi(arr: np.ndarray, mode: str, frac: float) -> tuple[np.ndarray, dict]:
    """ROI simples. `full` = imagem toda; `center` = recorte central `frac`.

    Para o pipeline real a ROI deve cair em tecido fibroglandular (evitar fundo
    de ar e a borda pele/músculo). Aqui é só um recorte grosseiro para o teste.
    """
    info = {"mode": mode, "frac": frac}
    if mode == "full":
        info["bbox"] = [0, 0, arr.shape[0], arr.shape[1]]
        return arr, info
    h, w = arr.shape
    rh, rw = int(h * frac), int(w * frac)
    r0, c0 = (h - rh) // 2, (w - rw) // 2
    info["bbox"] = [r0, c0, r0 + rh, c0 + rw]
    return arr[r0 : r0 + rh, c0 : c0 + rw], info


def quantize(arr: np.ndarray, levels: int, clip_pct: float) -> np.ndarray:
    """Reescala para [0, levels-1] usando percentis (robusto a outliers)."""
    lo, hi = np.percentile(arr, [clip_pct, 100 - clip_pct])
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max() + 1e-6)
    norm = np.clip((arr - lo) / (hi - lo), 0, 1)
    q = np.round(norm * (levels - 1)).astype(np.uint8)
    return q


def glcm_entropy(glcm: np.ndarray) -> np.ndarray:
    """Entropia de Shannon da GLCM, por (distância, ângulo)."""
    p = glcm.astype(np.float64)
    p /= p.sum(axis=(0, 1), keepdims=True) + 1e-12
    ent = -np.sum(p * np.log2(p + 1e-12), axis=(0, 1))
    return ent  # shape (n_dist, n_angle)


def compute(arr_q: np.ndarray, levels: int) -> dict:
    glcm = graycomatrix(
        arr_q,
        distances=DISTANCES,
        angles=ANGLES_RAD,
        levels=levels,
        symmetric=True,
        normed=False,  # graycoprops normaliza internamente; entropia normaliza à mão
    )
    out: dict = {"per_distance": {}}
    ent = glcm_entropy(glcm)

    for di, d in enumerate(DISTANCES):
        block: dict = {"per_angle": {}, "angle_mean": {}, "angle_std": {}}
        prop_vals = {}
        for prop in SKIMAGE_PROPS:
            vals = graycoprops(glcm, prop)[di]  # shape (n_angle,)
            prop_vals[prop] = vals
        prop_vals["entropy"] = ent[di]

        for ai, a in enumerate(ANGLES_DEG):
            block["per_angle"][f"{a}deg"] = {
                p: float(v[ai]) for p, v in prop_vals.items()
            }
        for p, v in prop_vals.items():
            block["angle_mean"][p] = float(np.mean(v))
            block["angle_std"][p] = float(np.std(v))
        out["per_distance"][f"d{d}"] = block
    return out


def print_report(meta: dict, roi_info: dict, levels: int, result: dict) -> None:
    print("=" * 72)
    print(f"imagem      : {meta['path']}")
    print(f"loader      : {meta['loader']}  shape={meta.get('shape')}")
    if meta.get("photometric"):
        print(f"photometric : {meta['photometric']}  bits_stored={meta.get('bits_stored')}")
    print(f"ROI         : {roi_info['mode']} frac={roi_info['frac']} bbox={roi_info['bbox']}")
    print(f"quantização : {levels} níveis | distâncias={DISTANCES} | ângulos={ANGLES_DEG}")
    print("=" * 72)

    props = SKIMAGE_PROPS + ["entropy"]
    for d in DISTANCES:
        block = result["per_distance"][f"d{d}"]
        print(f"\n[ distância = {d} px ]  (média ± desvio entre os 4 ângulos)")
        print("-" * 72)
        for p in props:
            m = block["angle_mean"][p]
            s = block["angle_std"][p]
            aniso = f"  aniso={s / (abs(m) + 1e-9):5.1%}" if p != "entropy" else ""
            print(f"  {p:<14} {m:12.5f}  ± {s:10.5f}{aniso}")
    print("\n(aniso = desvio/|média|: quão dependente da direção é a textura)")
    print("=" * 72)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path, help="PNG/JPG/TIF/DICOM")
    ap.add_argument("--levels", type=int, default=32, help="níveis de cinza para quantização (default 32)")
    ap.add_argument("--roi", choices=["full", "center"], default="center", help="default: center")
    ap.add_argument("--roi-frac", type=float, default=0.5, help="fração do lado no modo center (default 0.5)")
    ap.add_argument("--clip-pct", type=float, default=1.0, help="percentil de corte na normalização (default 1.0)")
    ap.add_argument("--json", type=Path, default=None, help="salva o resultado completo em JSON")
    ap.add_argument("--save-roi", type=Path, default=None, help="salva a ROI quantizada como PNG (sanity check)")
    args = ap.parse_args(argv)

    if not args.image.exists():
        print(f"erro: arquivo não encontrado: {args.image}", file=sys.stderr)
        return 2
    if args.levels < 2 or args.levels > 256:
        print("erro: --levels deve estar em [2, 256]", file=sys.stderr)
        return 2

    arr, meta = load_image(args.image)
    roi, roi_info = extract_roi(arr, args.roi, args.roi_frac)
    roi_q = quantize(roi, args.levels, args.clip_pct)

    if args.save_roi:
        from PIL import Image

        disp = (roi_q.astype(np.float64) / (args.levels - 1) * 255).astype(np.uint8)
        Image.fromarray(disp).save(args.save_roi)
        print(f"ROI quantizada salva em {args.save_roi}")

    result = compute(roi_q, args.levels)
    print_report(meta, roi_info, args.levels, result)

    if args.json:
        payload = {
            "meta": meta,
            "roi": roi_info,
            "params": {
                "levels": args.levels,
                "distances": DISTANCES,
                "angles_deg": ANGLES_DEG,
                "clip_pct": args.clip_pct,
            },
            "glcm": result,
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nJSON completo em {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
