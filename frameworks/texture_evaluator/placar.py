"""Placar por versão do gerador: a textura melhorou de uma iteração para a outra?

Cada versão é uma pasta com as sintéticas que uma iteração do modelo gerou a
partir das **mesmas originais** (mesmos nomes de arquivo). O placar:

1. calibra as réguas uma única vez, nas originais da versão de referência, e
   usa essas réguas em todas as versões — senão os números não se comparam;
2. avalia só os cortes válidos em todas as versões (mesmo conjunto de pares);
3. resume cada versão com a mediana de cada indicador e um intervalo de
   confiança de 95% por bootstrap nos cortes;
4. compara cada versão com a anterior, corte a corte: o indicador chegou mais
   perto do ideal? Só diz "melhorou" ou "piorou" quando o intervalo da
   diferença não cruza o zero e a mudança passa do mínimo relevante
   (`MIN_CHANGE`).

Uso:
    python -m frameworks.texture_evaluator.placar \\
        --version v1=CAMINHO/DA/VERSAO1 --version v2=CAMINHO/DA/VERSAO2 \\
        --mask-dir dados/Dgen --output-dir outputs/placar
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wavelet.src.load_data import CasePaths, discover_cases
from wavelet.src.run_pipeline import write_csv
from wavelet.src.validate_cases import ValidCase, validate_case
from frameworks.texture_evaluator.decision import FINAL_STATUSES
from frameworks.texture_evaluator.run_mvp import calibrate, evaluate_pair

BOOTSTRAP = 2000
# (coluna, nome, ideal, unidade): o ideal é a sintética igual à original
INDICATORS = (
    ("wavelet_fine_pct", "Detalhe fino (Wavelet)", 100.0, "% da original"),
    ("wavelet_coarse_pct", "Detalhe grosso (Wavelet)", 100.0, "% da original"),
    ("wavelet_ratio", "Razão fino/grosso", 1.0, "1 = sem borrar nem ruído"),
    ("glcm_contrast_pct", "Contraste GLCM", 100.0, "% da original"),
    ("delta_mu", "Δμ (brilho)", 0.0, "sintética − original"),
)
# menor mudança que conta como real (exploratório): abaixo disso, mesmo que o
# intervalo não cruze o zero, o placar diz "mudança pequena"
MIN_CHANGE = {"wavelet_fine_pct": 5.0, "wavelet_coarse_pct": 5.0, "wavelet_ratio": .05, "glcm_contrast_pct": 5.0, "delta_mu": .05, "aprovada": 5.0, "brilho": 5.0}
RATES = (
    ("aprovada", "Aprovadas", lambda row: row["final_status"] == "aprovada"),
    ("brilho", "Brilho sinalizado", lambda row: row["brightness_status"] != "approved" or row["density_status"] != "ok"),
)


def parse_version(text: str) -> tuple[str, Path]:
    name, _, folder = text.partition("=")
    if not folder:
        raise argparse.ArgumentTypeError("use NOME=PASTA, por exemplo v1=saidas/epoca50")
    return name, Path(folder)


def load_valid(folder: Path, mask_dir: Path | None) -> dict[str, ValidCase]:
    valid = {}
    for paths in discover_cases(folder):
        if mask_dir:
            paths = CasePaths(paths.case_id, paths.original, paths.synthetic, mask_dir / f"{paths.case_id}_mask.tiff")
        result = validate_case(paths, erosion_radius=2)
        if isinstance(result, ValidCase):
            valid[paths.case_id] = result
    return valid


def bootstrap(values: np.ndarray, statistic, rng) -> tuple[float, float, float]:
    """Estatística e intervalo de 95% reamostrando os cortes."""
    samples = [statistic(values[rng.integers(0, len(values), len(values))]) for _ in range(BOOTSTRAP)]
    low, high = np.percentile(samples, (2.5, 97.5))
    return float(statistic(values)), float(low), float(high)


def change_label(gain: float, low: float, high: float, key: str) -> str:
    if low <= 0 <= high:
        return "sem diferença clara"
    if abs(gain) < MIN_CHANGE[key]:
        return "mudança pequena"
    return "melhorou" if gain > 0 else "piorou"


def score(rows: list[dict], rng) -> dict:
    out = {}
    for key, _, _, _ in INDICATORS:
        out[key] = bootstrap(np.array([float(row[key]) for row in rows]), np.median, rng)
    for key, _, test in RATES:
        out[key] = tuple(100 * value for value in bootstrap(np.array([test(row) for row in rows], dtype=float), np.mean, rng))
    return out


def compare(previous: list[dict], current: list[dict], rng) -> dict:
    """Mudança corte a corte; positivo = a versão nova está mais perto do ideal."""
    out = {}
    for key, _, ideal, _ in INDICATORS:
        gain = np.array([abs(float(a[key]) - ideal) - abs(float(b[key]) - ideal) for a, b in zip(previous, current)])
        out[key] = bootstrap(gain, np.median, rng)
    for key, _, test in RATES:
        sign = 1 if key == "aprovada" else -1  # mais aprovadas é melhor; mais brilho sinalizado é pior
        gain = np.array([sign * (float(test(b)) - float(test(a))) for a, b in zip(previous, current)])
        out[key] = tuple(100 * value for value in bootstrap(gain, np.mean, rng))
    return out


def fmt(value: float, key: str) -> str:
    if key == "wavelet_ratio":
        return f"{value:.2f}"
    if key == "delta_mu":
        return f"{value:+.3f}"
    return f"{value:.0f}%"


def verdict_label(row: dict) -> str:
    """Veredito final com o motivo principal: brilho rejeita sozinho; em revisão, mostra o tipo de defeito."""
    if row["final_status"] == "rejeitar":
        return "rejeitar: brilho"
    if row["final_status"] == "revisar":
        return "revisar: só brilho" if row["wavelet_verdict"] == "textura preservada" else f"revisar: {row['wavelet_verdict']}"
    return "aprovada"


def write_markdown(names, scores, comparisons, verdicts, n_cases, dropped, output: Path) -> None:
    lines = [
        "# Placar por versão", "",
        f"{n_cases} cortes avaliados em todas as versões. Réguas calibradas nas originais de `{names[0]}`.",
        "Valor = mediana entre os cortes; entre colchetes, intervalo de 95% (bootstrap).", "",
        "## Indicadores", "",
        "| Indicador | Ideal | " + " | ".join(names) + " |",
        "|---|---|" + "---|" * len(names),
    ]
    for key, label, ideal, unit in INDICATORS:
        cells = [f"{fmt(v, key)} [{fmt(lo, key)}, {fmt(hi, key)}]" for v, lo, hi in (scores[name][key] for name in names)]
        lines.append(f"| {label} ({unit}) | {fmt(ideal, key)} | " + " | ".join(cells) + " |")
    for key, label, _ in RATES:
        cells = [f"{v:.0f}% [{lo:.0f}, {hi:.0f}]" for v, lo, hi in (scores[name][key] for name in names)]
        lines.append(f"| {label} (% dos cortes) | {'100%' if key == 'aprovada' else '0%'} | " + " | ".join(cells) + " |")

    lines += ["", "## Vereditos", "", "| Veredito | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for label in sorted({label for counts in verdicts.values() for label in counts}, key=lambda label: -sum(counts.get(label, 0) for counts in verdicts.values())):
        lines.append(f"| {label} | " + " | ".join(str(verdicts[name].get(label, 0)) for name in names) + " |")

    if comparisons:
        lines += [
            "", "## Comparação com a versão anterior", "",
            "Ganho = quanto, corte a corte, a versão nova chegou mais perto do ideal (mediana; em taxas, pontos percentuais).",
            "Só é \"melhorou\" ou \"piorou\" quando o intervalo de 95% não cruza o zero **e** a mudança passa do mínimo relevante",
            "(5 pp nas porcentagens e taxas, 0,05 na razão e no Δμ); se só a primeira condição vale, aparece \"mudança pequena\".", "",
            "| Indicador | " + " | ".join(f"{a} → {b}" for a, b in comparisons) + " |",
            "|---|" + "---|" * len(comparisons),
        ]
        for key, label in [(key, label) for key, label, *_ in INDICATORS] + [(key, label) for key, label, _ in RATES]:
            cells = []
            for pair in comparisons:
                gain, low, high = comparisons[pair][key]
                # razão e Δμ na própria unidade; porcentagens e taxas em pontos percentuais
                number = f"{gain:+.2f}" if key in ("wavelet_ratio", "delta_mu") else f"{gain:+.1f} pp"
                cells.append(f"**{change_label(gain, low, high, key)}** ({number})")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    if dropped:
        lines += ["", "## Cortes fora do placar", "", "Válidos em alguma versão, mas não em todas: " + ", ".join(f"`{case}`" for case in sorted(dropped)) + "."]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_figure(names, scores, output: Path) -> None:
    panels = INDICATORS[:4] + (("aprovada", "Aprovadas", 100.0, "% dos cortes"),)
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6))
    x = np.arange(len(names))
    for axis, (key, label, ideal, unit) in zip(axes, panels):
        values = np.array([scores[name][key] for name in names])
        axis.errorbar(x, values[:, 0], yerr=[values[:, 0] - values[:, 1], values[:, 2] - values[:, 0]], fmt="o-", color="#4C78A8", capsize=4)
        axis.axhline(ideal, color="#54A24B", linestyle="--", linewidth=1, label="ideal")
        axis.set_xticks(x, names, fontsize=9)
        axis.set_title(f"{label}\n({unit})", fontsize=9)
        axis.set_xlim(-.4, len(names) - .6)
    axes[0].legend(fontsize=8)
    fig.suptitle("Placar por versão: mediana e intervalo de 95% (linha verde = sintética igual à original)", fontsize=10)
    fig.tight_layout()
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", type=parse_version, action="append", required=True, help="NOME=PASTA, em ordem cronológica; repita para cada versão")
    p.add_argument("--mask-dir", type=Path, help="pasta das máscaras, se não estiverem junto das imagens")
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    names = [name for name, _ in a.version]
    if len(set(names)) != len(names):
        raise SystemExit("nomes de versão repetidos")
    missing = [f"  {name}={folder}" for name, folder in a.version if not folder.is_dir()]
    if missing:
        raise SystemExit("pasta de versão não encontrada:\n" + "\n".join(missing) + "\ncada --version deve apontar para uma pasta existente com as originais e as sintéticas (_generate)")
    if a.mask_dir and not a.mask_dir.is_dir():
        raise SystemExit(f"pasta de máscaras não encontrada: {a.mask_dir}")

    valid = {name: load_valid(folder, a.mask_dir) for name, folder in a.version}
    common = sorted(set.intersection(*(set(cases) for cases in valid.values())))
    if not common:
        raise SystemExit("nenhum corte válido em todas as versões")
    dropped = set.union(*(set(cases) for cases in valid.values())) - set(common)
    reference = valid[names[0]]
    # as originais precisam ser as mesmas em todas as versões, senão o placar compara coisas diferentes
    for name in names[1:]:
        for case in common:
            if not np.array_equal(valid[name][case].original, reference[case].original):
                raise SystemExit(f"a original de {case} difere entre {names[0]} e {name}")

    ctx = calibrate([reference[case] for case in common])
    rows = {name: [evaluate_pair(case, valid[name][case].original, valid[name][case].synthetic, valid[name][case].lung_mask, ctx) for case in common] for name in names}
    rng = np.random.default_rng(0)
    scores = {name: score(rows[name], rng) for name in names}
    comparisons = {(a_, b_): compare(rows[a_], rows[b_], rng) for a_, b_ in zip(names, names[1:])}
    verdicts = {name: Counter(verdict_label(row) for row in rows[name]) for name in names}

    a.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(a.output_dir / "por_caso.csv", [{"version": name, **row} for name in names for row in rows[name]])
    write_csv(a.output_dir / "placar.csv", [
        {"version": name, "indicator": key, "value": v, "ci_low": lo, "ci_high": hi}
        for name in names for key, (v, lo, hi) in scores[name].items()
    ])
    write_csv(a.output_dir / "comparacao.csv", [
        {"from": a_, "to": b_, "indicator": key, "gain": g, "ci_low": lo, "ci_high": hi, "result": change_label(g, lo, hi, key)}
        for (a_, b_), values in comparisons.items() for key, (g, lo, hi) in values.items()
    ])
    (a.output_dir / "protocolo.json").write_text(json.dumps({
        "versions": {name: str(folder) for name, folder in a.version}, "reference": names[0],
        "n_cases": len(common), "dropped_cases": sorted(dropped), "bootstrap": BOOTSTRAP,
        "glcm_scale": ctx.scale.as_dict(), "density_interval": ctx.interval, "density_limits": ctx.limits,
        "final_statuses": FINAL_STATUSES,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(names, scores, comparisons, verdicts, len(common), dropped, a.output_dir / "placar.md")
    save_figure(names, scores, a.output_dir / "placar.png")
    print((a.output_dir / "placar.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
