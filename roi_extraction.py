"""
roi_extraction.py — extração de ROI de tecido mamário, reutilizável por qualquer métrica.

Este módulo isola UMA responsabilidade: dado uma mamografia, achar um (ou mais)
recorte(s) quadrado(s) que caia(m) de forma confiável DENTRO do parênquima —
longe do fundo de ar, da linha de pele, de marcadores/texto e (opcionalmente) do
músculo peitoral. O recorte cru (intensidades originais) é devolvido para o
código da métrica fazer o que precisar (GLCM, Power Spectrum/NPS, wavelets, LBP…).

Por que um módulo separado: várias camadas do framework de validação precisam da
mesma ROI. Ter isso num lugar só, bem testado, evita cada métrica reinventar
(mal) a extração.

--------------------------------------------------------------------------------
USO COMO BIBLIOTECA
--------------------------------------------------------------------------------
    from roi_extraction import load_image, extract_roi

    img = load_image("caso.npz")           # .npz / .npy / .png|.jpg|.tif / .dcm
    res = extract_roi(img, size=128)

    if res.ok:
        roi = res.roi.array                # np.ndarray (size x size), intensidades cruas
        ...                                # entrega para a métrica
    else:
        print("rejeitada:", res.reject_reason)

    # várias ROIs (ex.: para média de NPS ou distribuições):
    res = extract_roi(img, size=256, n_rois=8)
    for r in res.rois:
        ...

--------------------------------------------------------------------------------
USO COMO FERRAMENTA (linha de comando)
--------------------------------------------------------------------------------
    python roi_extraction.py "2d_only_1000_images/images/*.npz" --preview
    python roi_extraction.py caso.npz --size 256 --n-rois 4 --preview --out previews/
    python roi_extraction.py "imgs/*.npz" --csv rois.csv      # só o relatório/CSV

Gera um PNG de conferência por imagem (imagem + contorno da mama + ROIs) e/ou um
CSV com posição, frações e o diagnóstico de qualidade de cada imagem.
"""
from __future__ import annotations

import os
import warnings as _warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_li, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, remove_small_objects

# skimage 0.26 emite FutureWarnings de deprecação em assinaturas que ainda
# funcionam (ex.: `min_size` de remove_small_objects). Silenciamos essas para
# não poluir a saída da ferramenta.
_warnings.filterwarnings("ignore", category=FutureWarning,
                         message=r".*(min_size|binary_closing|remove_small_objects).*")

# ---------------------------------------------------------------------------
# PADRÕES (ajustáveis por argumento em extract_roi / pela CLI)
# ---------------------------------------------------------------------------
DEFAULT_SIZE = 128            # lado do recorte quadrado, em pixels
MIN_TISSUE_FRACTION = 0.98    # fração mínima da ROI que precisa cair na mama
SKIN_MARGIN_FRAC = 0.15       # folga extra da borda da pele, como fração de `size`
BRIGHT_PERCENTILE = 99.6      # acima disto (dentro da mama) = marcador/calcificação/saturação
MAX_SATURATED_IN_ROI = 0.02   # fração máxima de pixels "muito claros" tolerada na ROI

# limites das checagens de sanidade da máscara (calibrados no OMAMA-DB)
# calibrados p/ REJEITAR espécime/plate/painel sem derrubar mamografia real
# (mama grande tem pouco ar; mama pode ter uma pequena margem da borda).
MIN_BG_FRACTION = 0.04        # abaixo disto: quase sem ar no quadro
MIN_LARGEST_COMP_FRACTION = 0.85  # maior componente / todo o "não-fundo": abaixo = máscara fragmentada
MAX_EXTENT = 0.93           # área / área da bounding box; acima disto = placa/painel retangular
MIN_AREA_FRAC = 0.06         # silhueta ocupa menos que isto do quadro = máscara fina/incorreta (limiar falhou)
MIN_EDGE_CONTACT = 0.02      # a mama é flush com a parede torácica; placa "flutuante" ~0
MAX_HOLES_NEG = -12          # euler_number do componente cru abaixo disto = placa de grade (dezenas de buracos)
WORK_MAX = 1100             # a máscara/elegível são calculadas nesta resolução máx. (o recorte final é full-res)
WORK_MAX = 1100             # a máscara/elegível são calculadas nesta resolução máx. (o recorte final é full-res)


# ---------------------------------------------------------------------------
# ESTRUTURAS DE RETORNO
# ---------------------------------------------------------------------------
@dataclass
class ROI:
    """Um recorte extraído."""
    array: np.ndarray                 # intensidades cruas, (size, size)
    bbox: tuple[int, int, int, int]   # (x0, y0, x1, y1)
    center: tuple[int, int]           # (cx, cy)
    tissue_fraction: float            # fração da ROI dentro da máscara da mama
    saturated_fraction: float         # fração de pixels acima do percentil de brilho
    mean: float
    std: float


@dataclass
class ROIResult:
    image: np.ndarray
    breast_mask: np.ndarray
    eligible_mask: np.ndarray
    rois: list[ROI] = field(default_factory=list)
    ok: bool = False
    warnings: list[str] = field(default_factory=list)
    rejected: bool = False
    reject_reason: str | None = None
    source: str | None = None
    mask_stats: dict = field(default_factory=dict)

    @property
    def roi(self) -> ROI | None:
        """A primeira (e normalmente única) ROI, ou None se não houver."""
        return self.rois[0] if self.rois else None


# ---------------------------------------------------------------------------
# CARREGAMENTO
# ---------------------------------------------------------------------------
def load_image(path: str) -> np.ndarray:
    """
    Carrega uma imagem 2D como float. Suporta:
      .npz  -> primeiro array salvo dentro
      .npy  -> o array
      .png/.jpg/.jpeg/.tif/.tiff -> via Pillow (convertida para tons de cinza)
      .dcm/.dicom -> via pydicom (aplica RescaleSlope/Intercept; corrige MONOCHROME1)
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npz":
        with np.load(path) as data:
            arr = data[data.files[0]]
    elif ext == ".npy":
        arr = np.load(path)
    elif ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        from PIL import Image
        im = Image.open(path)
        if im.mode not in {"I", "I;16", "F", "L"}:
            im = im.convert("L")
        arr = np.asarray(im)
    elif ext in {".dcm", ".dicom"}:
        import pydicom
        ds = pydicom.dcmread(path, force=True)
        arr = ds.pixel_array.astype(np.float64)
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
        arr = arr * slope + intercept
        if str(getattr(ds, "PhotometricInterpretation", "")).strip() == "MONOCHROME1":
            arr = arr.max() - arr
    else:
        raise ValueError(f"extensão não suportada: {ext!r} ({path})")

    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    if arr.ndim != 2:
        raise ValueError(f"esperava imagem 2D, veio shape {arr.shape}")
    return arr


# ---------------------------------------------------------------------------
# MÁSCARA DA MAMA (silhueta) + CHECAGENS DE SANIDADE
# ---------------------------------------------------------------------------
def _robust_background_threshold(img: np.ndarray) -> float:
    """
    Limiar que separa o fundo de ar do resto: **min(Otsu, Li)**.

    Nenhum método sozinho cobre os dois casos difíceis do OMAMA-DB:
      - **mama gordurosa** (interior escuro): Otsu cai ENTRE gordura e tecido
        denso e a máscara sai só do núcleo denso. Li senta mais baixo e pega a
        mama inteira.
      - **fundo NÃO preto** (exposição/gradiente de fundo elevado): um corte
        muito baixo pegaria o fundo todo como "tecido". Otsu e Li separam certo
        aqui (testado).
    Pegar o menor dos dois resolve o caso gorduroso sem estragar os demais.
    'Lua crescente' (borda densa só) é resolvida depois por _fill_from_chestwall.
    """
    lo, hi = np.percentile(img, [0.5, 99.9])
    cands = []
    for fn in (threshold_otsu, threshold_li):
        try:
            t = float(fn(img))
            if lo < t < hi:
                cands.append(t)
        except Exception:
            pass
    return min(cands) if cands else lo + 0.05 * (hi - lo)


def _fill_from_chestwall(comp: np.ndarray) -> np.ndarray:
    """
    Preenche a silhueta da mama a partir da parede torácica.

    Quando o interior gorduroso fica abaixo do limiar, o `img > thr` sai como
    uma 'lua crescente' (só a borda densa/pele) e o fill_holes não fecha nada,
    porque a região é aberta para o fundo. Aqui usamos a geometria da
    mamografia: a mama é sólida entre a parede torácica (a borda do quadro que
    ela mais toca) e a sua borda mais externa. Então, linha a linha (ou coluna
    a coluna), pinta tudo entre a parede e o pixel de tecido mais distante.
    """
    h, w = comp.shape
    touch = {"esq": comp[:, 0].sum(), "dir": comp[:, -1].sum(),
             "topo": comp[0, :].sum(), "baixo": comp[-1, :].sum()}
    parede = max(touch, key=touch.get)
    out = comp.copy()

    if parede in ("esq", "dir"):
        for y in range(h):
            xs = np.flatnonzero(comp[y])
            if xs.size:
                out[y, xs.min():xs.max() + 1] = True
    else:
        for x in range(w):
            ys = np.flatnonzero(comp[:, x])
            if ys.size:
                out[ys.min():ys.max() + 1, x] = True
    return out


def breast_silhouette(img: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Devolve (mask_mama, stats). A máscara é a silhueta preenchida do maior
    objeto contíguo (a mama), com fechamento morfológico e buracos tapados —
    então ela representa 'o que é mama', independente da variação de brilho
    interna.

    stats traz números usados nas checagens de sanidade:
      bg_fraction, solidity, largest_comp_fraction, n_components, area_frac
    """
    thr = _robust_background_threshold(img)
    raw = img > thr
    bg_level = float(np.percentile(img, 2))

    # limpeza morfológica proporcional ao tamanho da imagem
    r = max(1, min(img.shape) // 200)
    closed = closing(raw, disk(r))
    filled = ndi.binary_fill_holes(closed)
    filled = remove_small_objects(filled, min_size=max(64, img.size // 5000))

    lab = label(filled)
    stats = {
        "n_components": int(lab.max()),
        "bg_fraction": float(1.0 - raw.mean()),
    }
    if lab.max() == 0:
        return np.zeros_like(img, dtype=bool), {**stats, "solidity": 0.0,
                                                "largest_comp_fraction": 0.0,
                                                "area_frac": 0.0, "extent": 0.0,
                                                "edge_contact": 0.0, "euler": 1}

    # A mama é a maior região BRILHANTE. Ponderar por (área × brilho acima do
    # fundo) impede que um leve gradiente no fundo preto — que às vezes passa do
    # limiar e forma um blob grande e escuro — seja escolhido no lugar da mama.
    regs = regionprops(lab, intensity_image=img)
    def _peso(p):
        return p.area * max(p.mean_intensity - bg_level, 1.0)
    maior = max(regs, key=_peso)
    comp = lab == maior.label
    total_fg = float(filled.sum())
    stats["largest_comp_fraction"] = float(maior.area / total_fg) if total_fg else 0.0
    # nº de buracos no componente CRU (antes de tapar): placa de grade de
    # espécime tem dezenas (grade + letras); mama é simplesmente conexa (~1).
    stats["euler"] = int(maior.euler_number)

    # preenche a silhueta a partir da parede torácica (resolve a 'lua crescente'
    # quando o interior gorduroso fica abaixo do limiar) e tapa buracos internos.
    mask = ndi.binary_fill_holes(_fill_from_chestwall(comp))

    # solidity/extent recalculados NA MÁSCARA FINAL (não no componente cru)
    mp = regionprops(mask.astype(np.uint8))[0]
    stats["solidity"] = float(mp.solidity)
    stats["extent"] = float(mp.extent)
    stats["area_frac"] = float(mask.sum() / mask.size)
    # contato com as bordas do quadro: a mama fica flush com a parede torácica,
    # então PELO MENOS uma borda tem boa cobertura. Uma placa/espécime que
    # "flutua" no meio do quadro não encosta em borda nenhuma.
    stats["edge_contact"] = float(max(mask[0, :].mean(), mask[-1, :].mean(),
                                      mask[:, 0].mean(), mask[:, -1].mean()))
    return mask, stats


def _sanity_check(stats: dict) -> tuple[bool, str | None, list[str]]:
    """Traduz os stats da máscara em (rejeitar?, motivo, avisos)."""
    warnings: list[str] = []

    if stats.get("area_frac", 0) < MIN_AREA_FRAC:
        return True, (f"silhueta ocupa só {stats.get('area_frac', 0):.1%} do quadro — "
                      f"máscara fina/incorreta (limiar provavelmente falhou)"), warnings

    if stats.get("edge_contact", 1.0) < MIN_EDGE_CONTACT:
        return True, (f"máscara não encosta em nenhuma borda do quadro "
                      f"({stats.get('edge_contact', 0):.0%}) — não é mama flush com a "
                      f"parede torácica (espécime em placa, painel, artefato)"), warnings

    if stats["bg_fraction"] < MIN_BG_FRACTION:
        return True, (f"pouco fundo de ar ({stats['bg_fraction']:.0%}) — provável "
                      f"radiografia de espécime/plate, não mamografia in vivo"), warnings

    if stats.get("euler", 1) < MAX_HOLES_NEG:
        return True, (f"componente com muitos buracos (euler={stats['euler']}) — "
                      f"placa de grade de espécime, não silhueta de mama"), warnings

    if stats["largest_comp_fraction"] < MIN_LARGEST_COMP_FRACTION:
        return True, (f"máscara fragmentada (maior componente = "
                      f"{stats['largest_comp_fraction']:.0%} do não-fundo) — "
                      f"limiar/processamento atípico"), warnings

    # OBS: solidity NÃO é critério de rejeição — mama normal (sobretudo CC e a
    # curva axilar/inframamária em MLO) é francamente côncava do lado da parede
    # torácica, com solidity 0.5–0.8. Fica só como aviso.

    if stats.get("extent", 0) > MAX_EXTENT:
        return True, (f"maior região preenche {stats['extent']:.0%} da própria "
                      f"bounding box — retângulo/placa (espécime, painel, "
                      f"colimação), não silhueta de mama"), warnings

    if stats.get("extent", 0) > 0.85:
        warnings.append(f"silhueta bem retangular (extent={stats['extent']:.2f}) — "
                        f"conferir se não é espécime/painel")
    if stats["area_frac"] > 0.92:
        warnings.append(f"mama ocupa {stats['area_frac']:.0%} do quadro — "
                        f"pouco ar; conferir se é mamografia padrão")
    if stats["solidity"] < 0.55:
        warnings.append(f"silhueta bem côncava (solidity={stats['solidity']:.2f}) — "
                        f"normal em CC; conferir a máscara no preview")
    return False, None, warnings


# ---------------------------------------------------------------------------
# MÁSCARA "ELEGÍVEL" (onde o CENTRO da ROI pode cair)
# ---------------------------------------------------------------------------
def _pectoral_mask(img: np.ndarray, breast: np.ndarray) -> np.ndarray:
    """
    Heurística CONSERVADORA para o músculo peitoral (aparece em incidências
    MLO como um triângulo claro colado num canto SUPERIOR, tocando a borda
    lateral). Só marca algo se houver uma região claramente clara e grande
    encostada num canto de cima. Nunca remove da silhueta da mama — só da
    área elegível para o centro da ROI.
    """
    h, w = img.shape
    ys, xs = np.where(breast)
    if len(xs) == 0:
        return np.zeros_like(breast)
    p80 = np.percentile(img[breast], 80)
    bright = (img > p80) & breast

    out = np.zeros_like(breast)
    banda = slice(0, max(1, h // 3))  # terço superior
    for lado in ("esq", "dir"):
        col = slice(0, max(1, w // 3)) if lado == "esq" else slice(w - max(1, w // 3), w)
        canto = np.zeros_like(breast)
        canto[banda, col] = bright[banda, col]
        canto = remove_small_objects(canto, min_size=img.size // 400)
        if not canto.any():
            continue
        lab = label(canto)
        for p in regionprops(lab):
            miny, minx, maxy, maxx = p.bbox
            toca_topo = miny <= h * 0.02
            toca_lado = (minx <= w * 0.02) or (maxx >= w * 0.98)
            if toca_topo and toca_lado and p.area > breast.sum() * 0.02:
                comp = lab == p.label
                # fecho convexo do componente ~ triângulo do peitoral
                out |= ndi.binary_fill_holes(comp)
    return out


def _compact_bright_blobs(img, breast, size):
    """
    Pixels muito claros que pertencem a componentes PEQUENOS e COMPACTOS —
    marcadores, BBs de localização, calcificações focais. Tecido denso difuso
    forma regiões grandes e irregulares e NÃO entra aqui (senão a máscara
    elegível viraria queijo suíço em mamas densas).
    """
    vals = img[breast]
    if not vals.size:
        return np.zeros_like(breast)
    # semente bem restritiva: só o topo do brilho (marcador/BB costuma saturar)
    hot = (img > np.percentile(vals, 99.9)) & breast
    hot = remove_small_objects(hot, min_size=8)
    lab = label(hot)
    out = np.zeros_like(breast)
    max_area = min(0.003 * breast.sum(), (3 * size) ** 2)  # marcador: pequeno
    for p in regionprops(lab):
        if p.area <= max_area and p.solidity >= 0.75:
            out |= lab == p.label
    if out.any():
        out = ndi.binary_dilation(out, iterations=max(2, size // 16))
    return out


def _eligible_mask(img, breast, size, skin_margin_px, exclude_bright, exclude_pectoral):
    """
    Onde o CENTRO da ROI pode cair para o recorte sair 100% dentro da mama e
    longe da pele/marcadores/peitoral.

    Base: silhueta erodida por (size/2 + folga) via distância chebyshev — rápido
    e garante o quadrado inteiro dentro. Só DEPOIS retira marcadores compactos
    e (opcional) o peitoral, e apenas se sobrar região.
    """
    warnings: list[str] = []
    raio = size // 2 + int(skin_margin_px)
    cdt = ndi.distance_transform_cdt(breast, metric="chessboard")
    elig = cdt >= raio
    if not elig.any():
        return elig, warnings

    base = int(elig.sum())

    if exclude_pectoral:
        pect = _pectoral_mask(img, breast)
        novo = elig & ~pect
        if pect.any() and novo.sum() >= 0.30 * base:
            elig = novo
            warnings.append("possível músculo peitoral detectado e evitado (MLO)")

    if exclude_bright:
        blobs = _compact_bright_blobs(img, breast, size)
        novo = elig & ~blobs
        if blobs.any():
            if novo.sum() >= 0.60 * base:
                elig = novo
            else:
                warnings.append("muitos focos muito claros na mama — NÃO excluídos "
                                "(provável tecido denso, não marcador); conferir ROI")

    return elig, warnings


# ---------------------------------------------------------------------------
# EXTRAÇÃO
# ---------------------------------------------------------------------------
def _upscale_mask(m: np.ndarray, shape: tuple[int, int], scale: int) -> np.ndarray:
    """Volta uma máscara calculada em escala reduzida para o tamanho original."""
    if scale == 1:
        return m
    big = np.repeat(np.repeat(m, scale, axis=0), scale, axis=1)
    return big[: shape[0], : shape[1]]


def _crop_bbox(cx, cy, size, shape):
    H, W = shape
    x0 = int(np.clip(cx - size // 2, 0, W - size))
    y0 = int(np.clip(cy - size // 2, 0, H - size))
    return x0, y0, x0 + size, y0 + size


def _make_roi(img, breast, hot_thr, cx, cy, size):
    x0, y0, x1, y1 = _crop_bbox(cx, cy, size, img.shape)
    arr = img[y0:y1, x0:x1]
    m = breast[y0:y1, x0:x1]
    sat = float(np.mean(arr > hot_thr)) if hot_thr is not None else 0.0
    return ROI(array=arr, bbox=(x0, y0, x1, y1), center=(int(cx), int(cy)),
              tissue_fraction=float(m.mean()), saturated_fraction=sat,
              mean=float(arr.mean()), std=float(arr.std()))


def extract_roi(
    image: np.ndarray,
    size: int = DEFAULT_SIZE,
    n_rois: int = 1,
    *,
    min_tissue_fraction: float = MIN_TISSUE_FRACTION,
    skin_margin_frac: float = SKIN_MARGIN_FRAC,
    exclude_bright: bool = True,
    exclude_pectoral: bool = True,
    source: str | None = None,
) -> ROIResult:
    """
    Acha `n_rois` recorte(s) `size`x`size` dentro do parênquima.

    Estratégia:
      1. Silhueta da mama (limiar robusto + morfologia + maior componente).
      2. Checagens de sanidade da máscara -> pode REJEITAR a imagem
         (espécime/plate, máscara fragmentada, silhueta não-convexa…).
      3. Máscara elegível = silhueta erodida por (size/2 + folga da pele),
         menos marcadores compactos e (opcional) o peitoral — mas só se sobrar
         região suficiente (senão mantém e avisa).
      4. 1ª ROI = centróide da região elegível (parênquima central, estável e
         reprodutível). ROIs extras por amostragem de ponto-mais-distante.
      5. Se a elegível ficar vazia, afrouxa a folga e depois cai para uma busca
         em grade por maior fração de tecido. Se ainda assim nada servir, REJEITA.

    Retorna um ROIResult (ver docstring do módulo).
    """
    img = np.asarray(image, dtype=np.float64)
    H, W = img.shape

    # Máscara e elegível são calculadas numa versão reduzida (rápido); o recorte
    # final sai da imagem em resolução plena.
    scale = max(1, int(np.ceil(max(H, W) / WORK_MAX)))
    small = img[::scale, ::scale] if scale > 1 else img
    size_s = max(8, int(round(size / scale)))

    breast_s, stats = breast_silhouette(small)
    res = ROIResult(image=img,
                    breast_mask=_upscale_mask(breast_s, (H, W), scale),
                    eligible_mask=np.zeros((H, W), dtype=bool),
                    source=source, mask_stats=stats)

    rejeitar, motivo, avisos = _sanity_check(stats)
    res.warnings.extend(avisos)
    if rejeitar:
        res.rejected, res.reject_reason = True, motivo
        return res

    skin_px_s = skin_margin_frac * size_s
    elig_s, avisos = _eligible_mask(small, breast_s, size_s, skin_px_s,
                                    exclude_bright, exclude_pectoral)
    res.warnings.extend(avisos)
    if not elig_s.any():
        elig_s, _ = _eligible_mask(small, breast_s, size_s, 0,
                                   exclude_bright, exclude_pectoral)
        if elig_s.any():
            res.warnings.append("folga da pele reduzida a 0 para achar ROI")

    res.eligible_mask = _upscale_mask(elig_s, (H, W), scale)
    breast = res.breast_mask
    vals = img[breast]
    hot_thr = float(np.percentile(vals, BRIGHT_PERCENTILE)) if vals.size else None

    if elig_s.any():
        ys, xs = np.where(elig_s)                     # coords na escala reduzida

        def _snap(cy, cx):
            """centro -> pixel elegível mais próximo (elig pode ser não-convexa)."""
            if 0 <= int(cy) < elig_s.shape[0] and 0 <= int(cx) < elig_s.shape[1] \
                    and elig_s[int(cy), int(cx)]:
                return int(cy), int(cx)
            i = np.argmin((ys - cy) ** 2 + (xs - cx) ** 2)
            return int(ys[i]), int(xs[i])

        # 1ª ROI: centróide da região elegível — parênquima central, estável e
        # reprodutível (o que importa para comparar real vs. sintético).
        centros = [_snap(ys.mean(), xs.mean())]
        # ROIs extras: amostragem por ponto-mais-distante, centros afastados >= size.
        while len(centros) < max(1, n_rois):
            d = np.full(len(ys), np.inf)
            for (ccy, ccx) in centros:
                d = np.minimum(d, np.hypot(ys - ccy, xs - ccx))
            j = int(np.argmax(d))
            if d[j] < size_s:
                break
            centros.append((int(ys[j]), int(xs[j])))

        rois: list[ROI] = []
        for (cy_s, cx_s) in centros:
            cy = int(round((cy_s + 0.5) * scale))     # volta para full-res
            cx = int(round((cx_s + 0.5) * scale))
            roi = _make_roi(img, breast, hot_thr, cx, cy, size)
            if roi.saturated_fraction > MAX_SATURATED_IN_ROI and not rois:
                res.warnings.append(
                    f"ROI escolhida tem {roi.saturated_fraction:.0%} de pixels "
                    f"muito claros (calcificação/marcador?)")
            rois.append(roi)

        res.rois = rois
        res.ok = len(rois) > 0
        if len(rois) < n_rois:
            res.warnings.append(f"pedidas {n_rois} ROIs, achei {len(rois)} "
                                f"não-sobrepostas dentro da região elegível")
        return res

    # fallback: busca em grade por maior fração de tecido (comportamento antigo)
    ys, xs = np.where(breast)
    x_lo, x_hi = xs.min() + size // 2, xs.max() - size // 2
    y_lo, y_hi = ys.min() + size // 2, ys.max() - size // 2
    if x_hi <= x_lo or y_hi <= y_lo:
        res.rejected, res.reject_reason = True, "mama menor que a ROI pedida"
        return res

    depth = ndi.distance_transform_edt(breast)
    melhor, melhor_key = None, None
    for cy in np.linspace(y_lo, y_hi, 15):
        for cx in np.linspace(x_lo, x_hi, 15):
            roi = _make_roi(img, breast, hot_thr, cx, cy, size)
            # desempata pela profundidade no centro (evita puxar p/ o canto da varredura)
            key = (round(roi.tissue_fraction, 3), float(depth[int(cy), int(cx)]))
            if melhor is None or key > melhor_key:
                melhor, melhor_key = roi, key
    if melhor is None or melhor.tissue_fraction < min_tissue_fraction:
        res.rejected = True
        res.reject_reason = (
            f"nenhuma ROI com fração de tecido >= {min_tissue_fraction:.0%} "
            f"(melhor: {melhor.tissue_fraction:.0%} se houver)")
        return res

    res.warnings.append("ROI achada só pela busca em grade (elegível erodida "
                        "ficou vazia) — mama pequena/fina; conferir")
    res.rois = [melhor]
    res.ok = True
    return res


# ---------------------------------------------------------------------------
# CONVENIÊNCIA
# ---------------------------------------------------------------------------
def extract_roi_from_path(path: str, size: int = DEFAULT_SIZE, **kw) -> ROIResult:
    return extract_roi(load_image(path), size=size, source=os.path.basename(path), **kw)


def save_preview(res: ROIResult, out_path: str, titulo: str | None = None) -> None:
    """PNG de conferência: imagem + contorno da mama (ciano) + elegível (amarelo) + ROIs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from skimage.measure import find_contours

    fig, ax = plt.subplots(figsize=(6, 7), dpi=130)
    ax.imshow(res.image, cmap="gray")
    for c in find_contours(res.breast_mask.astype(float), 0.5):
        ax.plot(c[:, 1], c[:, 0], color="#00E5FF", lw=0.8)
    if res.eligible_mask.any():
        for c in find_contours(res.eligible_mask.astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], color="#FFD400", lw=0.8, alpha=0.8)
    for r in res.rois:
        x0, y0, x1, y1 = r.bbox
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                     edgecolor="#39FF14", lw=2))
    estado = ("REJEITADA: " + (res.reject_reason or "") if res.rejected
              else ("OK" if res.ok else "SEM ROI"))
    cab = titulo or res.source or ""
    ax.set_title(f"{cab}\n{estado}"
                 + (f"  | avisos: {len(res.warnings)}" if res.warnings else ""),
                 fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli(argv=None):
    import argparse
    import csv as _csv
    import glob as _glob

    ap = argparse.ArgumentParser(
        description="Extrai ROI de tecido mamário (reutilizável por qualquer métrica).")
    ap.add_argument("paths", nargs="+", help="arquivo(s) ou glob (.npz/.npy/.png/.dcm…)")
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE, help=f"lado da ROI (default {DEFAULT_SIZE})")
    ap.add_argument("--n-rois", type=int, default=1, help="quantas ROIs por imagem (default 1)")
    ap.add_argument("--no-exclude-bright", action="store_true", help="não evitar marcadores/calcificações claras")
    ap.add_argument("--no-exclude-pectoral", action="store_true", help="não tentar evitar o músculo peitoral")
    ap.add_argument("--preview", action="store_true", help="salvar PNG de conferência por imagem")
    ap.add_argument("--out", default="roi_previews", help="pasta dos previews (default roi_previews/)")
    ap.add_argument("--csv", default=None, help="salvar relatório CSV")
    args = ap.parse_args(argv)

    arquivos: list[str] = []
    for p in args.paths:
        hits = _glob.glob(p)
        arquivos.extend(sorted(hits) if hits else [p])
    if not arquivos:
        ap.error("nenhum arquivo encontrado")

    linhas = []
    n_ok = n_rej = n_warn = 0
    for i, path in enumerate(arquivos, 1):
        try:
            res = extract_roi_from_path(
                path, size=args.size, n_rois=args.n_rois,
                exclude_bright=not args.no_exclude_bright,
                exclude_pectoral=not args.no_exclude_pectoral)
        except Exception as e:
            print(f"[{i}/{len(arquivos)}] {os.path.basename(path)}: ERRO — {e}")
            linhas.append({"arquivo": os.path.basename(path), "estado": "erro",
                           "motivo": str(e)})
            continue

        if res.rejected:
            n_rej += 1
            estado = "rejeitada"
        elif res.ok:
            n_ok += 1
            estado = "ok"
        else:
            estado = "sem_roi"
        if res.warnings:
            n_warn += 1

        r0 = res.roi
        linhas.append({
            "arquivo": os.path.basename(path),
            "estado": estado,
            "motivo": res.reject_reason or "",
            "n_rois": len(res.rois),
            "bbox": ";".join(map(str, r0.bbox)) if r0 else "",
            "tissue_fraction": f"{r0.tissue_fraction:.3f}" if r0 else "",
            "saturated_fraction": f"{r0.saturated_fraction:.3f}" if r0 else "",
            "bg_fraction": f"{res.mask_stats.get('bg_fraction', 0):.3f}",
            "solidity": f"{res.mask_stats.get('solidity', 0):.3f}",
            "largest_comp_fraction": f"{res.mask_stats.get('largest_comp_fraction', 0):.3f}",
            "avisos": " | ".join(res.warnings),
        })

        marca = {"ok": "  ", "rejeitada": "REJ", "sem_roi": "!!", "erro": "ERR"}[estado]
        print(f"[{i}/{len(arquivos)}] {marca} {os.path.basename(path)}"
              + (f"  ({res.reject_reason})" if res.rejected else "")
              + (f"  [{len(res.warnings)} aviso(s)]" if res.warnings else ""))

        if args.preview:
            nome = os.path.splitext(os.path.basename(path))[0]
            save_preview(res, os.path.join(args.out, nome + ".png"))

    print(f"\n{len(arquivos)} imagens | ok={n_ok}  rejeitadas={n_rej}  "
          f"com avisos={n_warn}")
    if args.preview:
        print(f"previews em {args.out}/")

    if args.csv:
        campos = ["arquivo", "estado", "motivo", "n_rois", "bbox", "tissue_fraction",
                  "saturated_fraction", "bg_fraction", "solidity",
                  "largest_comp_fraction", "avisos"]
        with open(args.csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=campos)
            w.writeheader()
            for ln in linhas:
                w.writerow({k: ln.get(k, "") for k in campos})
        print(f"CSV em {args.csv}")


if __name__ == "__main__":
    _cli()
