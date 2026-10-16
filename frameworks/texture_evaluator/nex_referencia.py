"""Referência real do NEx na ROI do MVP.

O NEx (`wavelet/src/nyquist_excess.py`) foi caracterizado na imagem inteira com
o fundo harmonizado (ver `resultados/origem_assinatura_nex/`). O MVP entrega a
cada métrica a ROI pulmonar erodida (raio 2), então a faixa real precisa ser
medida nessa mesma ROI antes de o NEx entrar no avaliador.

Medida por imagem (fixada antes de olhar os resultados):

- janelas quadradas de 32 px, passo 16 px, **inteiramente dentro** da ROI
  erodida (borda da máscara fora: ela contamina o espectro);
- NEx de cada janela; o valor da imagem é a **mediana** das janelas;
- menos de 5 janelas → flag `poucas_janelas` (o caso fica na tabela).

Sensibilidade: a mesma medida com janelas ≥ 90 % dentro da ROI.

Saídas: faixa das originais (a referência), cada versão sintética contra ela
(Δ pareado e fração fora da faixa) e, para comparar, o NEx da imagem inteira.

    .venv/bin/python -m frameworks.texture_evaluator.nex_referencia \\
      --version Dsr=dados/Dsr --version Dgen=dados/Dgen --mask-dir dados/Dgen \\
      --output-dir resultados/nex_referencia_roi
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

from wavelet.src.load_data import CasePaths, discover_cases
from wavelet.src.nyquist_excess import nyquist_excess
from wavelet.src.run_pipeline import write_csv
from wavelet.src.validate_cases import ValidCase, validate_case

TILE, STRIDE, MIN_TILES = 32, 16, 5


def nex_roi(image: np.ndarray, roi: np.ndarray, coverage: float = 1.0) -> tuple[float, int]:
    """Mediana do NEx nas janelas TILE×TILE com pelo menos `coverage` da área na ROI."""
    values = [
        nyquist_excess(image[r:r + TILE, c:c + TILE])
        for r in range(0, image.shape[0] - TILE + 1, STRIDE)
        for c in range(0, image.shape[1] - TILE + 1, STRIDE)
        if roi[r:r + TILE, c:c + TILE].mean() >= coverage
    ]
    return (float(np.median(values)) if values else float("nan")), len(values)


def measure(image: np.ndarray, roi: np.ndarray) -> dict:
    roi100, n100 = nex_roi(image, roi, 1.0)
    roi90, n90 = nex_roi(image, roi, 0.9)
    return {"nex_roi": roi100, "janelas": n100, "poucas_janelas": n100 < MIN_TILES,
            "nex_roi_90": roi90, "janelas_90": n90, "nex_imagem_inteira": nyquist_excess(image)}


def load_version(data_dir: Path, mask_dir: Path | None) -> list[ValidCase]:
    cases = []
    for paths in discover_cases(data_dir):
        if mask_dir:
            paths = CasePaths(paths.case_id, paths.original, paths.synthetic, mask_dir / f"{paths.case_id}_mask.tiff")
        result = validate_case(paths, erosion_radius=2)
        if isinstance(result, ValidCase):
            cases.append(result)
    return cases


def summary(values: np.ndarray) -> dict:
    p5, q1, med, q3, p95 = np.percentile(values, [5, 25, 50, 75, 95])
    return {"n": int(values.size), "mediana": med, "q1": q1, "q3": q3, "iqr": q3 - q1, "p5": p5, "p95": p95}


def compare(real: np.ndarray, syn: np.ndarray, ref: dict) -> dict:
    """Sintética contra a faixa real: Δ pareado (mesmo corte) e posição na faixa."""
    delta = syn - real
    return {
        **{f"sint_{k}": v for k, v in summary(syn).items() if k != "n"},
        "delta_mediano": float(np.median(delta)),
        "delta_mediano_em_iqr": float(np.median(delta) / ref["iqr"]),
        "pares_acima_do_real": int((delta > 0).sum()),
        "wilcoxon_p": float(wilcoxon(delta).pvalue),
        "auc_real_vs_sint": float(mannwhitneyu(syn, real).statistic / (real.size * syn.size)),
        "frac_acima_p95_real": float((syn > ref["p95"]).mean()),
        "frac_abaixo_p5_real": float((syn < ref["p5"]).mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="append", required=True, help="NOME=PASTA (original + _generate)")
    p.add_argument("--mask-dir", type=Path, help="Pasta das máscaras, quando a versão não as tem")
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    rows, originals = [], {}
    for spec in a.version:
        name, folder = spec.split("=", 1)
        folder = Path(folder)
        mask_dir = None if any(folder.glob("*_mask.tiff")) else a.mask_dir
        for c in load_version(folder, mask_dir):
            if c.paths.case_id not in originals:
                originals[c.paths.case_id] = measure(c.original, c.lung_mask)
            rows.append({"versao": name, "case_id": c.paths.case_id,
                         **{f"real_{k}": v for k, v in originals[c.paths.case_id].items()},
                         **{f"sint_{k}": v for k, v in measure(c.synthetic, c.lung_mask).items()}})
    write_csv(a.output_dir / "por_caso.csv", rows)

    result = {"protocolo": {"janela_px": TILE, "passo_px": STRIDE, "cobertura_roi": 1.0, "min_janelas": MIN_TILES,
                            "agregacao": "mediana das janelas", "roi": "pulmão (máscara = 1) erodido com raio 2, como no MVP",
                            "sensibilidade": "mesma medida com janelas >= 90 % na ROI"},
              "referencia_real": {}, "versoes": {}}
    real_ids = sorted(originals)
    for key in ("nex_roi", "nex_roi_90", "nex_imagem_inteira"):
        result["referencia_real"][key] = summary(np.array([originals[i][key] for i in real_ids]))
    result["referencia_real"]["poucas_janelas"] = sum(originals[i]["poucas_janelas"] for i in real_ids)
    result["referencia_real"]["janelas_mediana"] = float(np.median([originals[i]["janelas"] for i in real_ids]))
    for name in dict.fromkeys(r["versao"] for r in rows):
        sub = [r for r in rows if r["versao"] == name]
        result["versoes"][name] = {
            key: compare(np.array([r[f"real_{key}"] for r in sub]), np.array([r[f"sint_{key}"] for r in sub]),
                         result["referencia_real"][key])
            for key in ("nex_roi", "nex_roi_90", "nex_imagem_inteira")
        } | {"pares": len(sub), "poucas_janelas_sint": sum(r["sint_poucas_janelas"] for r in sub)}
    (a.output_dir / "resumo.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    print(json.dumps(result, indent=1, ensure_ascii=False, default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    main()
