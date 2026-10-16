"""Placar por versão do gerador: a textura melhorou de uma iteração para a outra?

Cada versão é uma pasta com as sintéticas que uma iteração do modelo gerou a
partir das **mesmas originais** (mesmos nomes de arquivo). O placar guarda um
**histórico** numa pasta: cada versão é avaliada uma única vez e versões novas
podem ser adicionadas a qualquer momento.

1. A primeira versão adicionada vira a referência: as réguas são calibradas nas
   originais dela e guardadas no histórico. Toda versão seguinte é medida com
   essas mesmas réguas, e as originais dela precisam ser idênticas às da
   referência — senão os números não se comparam.
2. O placar usa só os cortes válidos em todas as versões do histórico.
3. Cada indicador sai como mediana entre os cortes, com intervalo de 95% por
   bootstrap.
4. Cada versão é comparada com a anterior, corte a corte: o indicador chegou
   mais perto do ideal? Só diz "melhorou" ou "piorou" quando o intervalo da
   diferença não cruza o zero e a mudança passa do mínimo relevante
   (`MIN_CHANGE`).

Uso (adiciona versões ao histórico e regera o placar; versões já avaliadas não
são recalculadas):
    python -m frameworks.texture_evaluator.placar --output-dir outputs/placar \\
        --version v1=CAMINHO/DA/VERSAO1 --version v2=CAMINHO/DA/VERSAO2 --mask-dir dados/Dgen

Mais tarde, só a versão nova:
    python -m frameworks.texture_evaluator.placar --output-dir outputs/placar --version v3=CAMINHO/DA/VERSAO3

Interface: `streamlit run frameworks/texture_evaluator/ui_placar.py`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wavelet.src.load_data import CasePaths, discover_cases
from wavelet.src.run_pipeline import write_csv
from wavelet.src.validate_cases import ValidCase, validate_case
from frameworks.texture_evaluator.run_mvp import calibrate, context_from_dict, context_to_dict, evaluate_pair

BOOTSTRAP = 2000
HISTORY_FILE = "historico.json"
# (coluna, nome, ideal, unidade): o ideal é a sintética igual à original
INDICATORS = (
    ("wavelet_fine_pct", "Detalhe fino (Wavelet)", 100.0, "% da original"),
    ("wavelet_coarse_pct", "Detalhe grosso (Wavelet)", 100.0, "% da original"),
    ("wavelet_ratio", "Razão fino/grosso", 1.0, "1 = sem borrar nem ruído"),
    ("glcm_contrast_pct", "Contraste GLCM", 100.0, "% da original"),
    ("abs_delta_mu", "|Δμ| (desvio de brilho)", 0.0, "|sintética − original|"),
)
RATES = (
    ("aprovada", "Aprovadas", 100.0, lambda row: row["final_status"] == "aprovada"),
    ("brilho", "Brilho sinalizado", 0.0, lambda row: row["brightness_status"] != "approved" or row["density_status"] != "ok"),
)
# menor mudança que conta como real (exploratório): abaixo disso, mesmo que o
# intervalo não cruze o zero, o placar diz "mudança pequena"
MIN_CHANGE = {"wavelet_fine_pct": 5.0, "wavelet_coarse_pct": 5.0, "wavelet_ratio": .05, "glcm_contrast_pct": 5.0, "abs_delta_mu": .05, "aprovada": 5.0, "brilho": 5.0}


class PlacarError(Exception):
    """Erro de uso (pasta errada, originais diferentes...), mostrado sem traceback."""


# ---------------------------------------------------------------- histórico

def load_history(folder: Path) -> dict | None:
    path = folder / HISTORY_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_history(folder: Path, history: dict) -> None:
    (folder / HISTORY_FILE).write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def rows_path(folder: Path, name: str) -> Path:
    return folder / "versoes" / (re.sub(r"[^\w.-]+", "_", name) + ".csv")


def load_rows(folder: Path, name: str) -> list[dict]:
    with rows_path(folder, name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_valid(folder: Path, mask_dir: Path | None) -> dict[str, ValidCase]:
    valid = {}
    for paths in discover_cases(folder):
        if mask_dir:
            paths = CasePaths(paths.case_id, paths.original, paths.synthetic, mask_dir / f"{paths.case_id}_mask.tiff")
        result = validate_case(paths, erosion_radius=2)
        if isinstance(result, ValidCase):
            valid[paths.case_id] = result
    return valid


def fingerprint(image: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(image, dtype=np.float64).tobytes()).hexdigest()


def add_version(history_dir: Path, name: str, folder: Path, mask_dir: Path | None = None) -> str:
    """Avalia uma versão e a guarda no histórico. Devolve um resumo em texto."""
    name, folder = name.strip(), Path(folder)
    if not name:
        raise PlacarError("o nome da versão está vazio")
    if not folder.is_dir():
        raise PlacarError(f"pasta de versão não encontrada: {folder}\nela deve conter as originais e as sintéticas (_generate)")
    history = load_history(history_dir)
    if history and any(version["name"] == name for version in history["versions"]):
        raise PlacarError(f"já existe uma versão chamada {name!r} no histórico")
    mask_dir = Path(mask_dir) if mask_dir else (Path(history["mask_dir"]) if history and history["mask_dir"] else None)
    if mask_dir and not mask_dir.is_dir():
        raise PlacarError(f"pasta de máscaras não encontrada: {mask_dir}")

    valid = load_valid(folder, mask_dir)
    if not valid:
        raise PlacarError(f"nenhum par válido em {folder}")
    if history is None:
        # primeira versão: as originais dela definem as réguas de todo o histórico
        ctx = calibrate(list(valid.values()))
        history = {
            "reference": name, "mask_dir": str(mask_dir) if mask_dir else None,
            "calibration": context_to_dict(ctx),
            "originals": {case: fingerprint(c.original) for case, c in valid.items()},
            "versions": [],
        }
    ctx = context_from_dict(history["calibration"])
    known = history["originals"]
    different = [case for case, c in valid.items() if case in known and fingerprint(c.original) != known[case]]
    if different:
        raise PlacarError(f"{len(different)} originais diferem das da referência ({history['reference']}), por exemplo {different[0]}; o placar só compara versões geradas das mesmas originais")
    cases = sorted(case for case in valid if case in known)
    if not cases:
        raise PlacarError(f"nenhum corte de {folder} existe na referência ({history['reference']})")

    rows = []
    for case in cases:
        row = evaluate_pair(case, valid[case].original, valid[case].synthetic, valid[case].lung_mask, ctx)
        rows.append({**row, "abs_delta_mu": abs(row["delta_mu"])})
    history_dir.mkdir(parents=True, exist_ok=True)
    rows_path(history_dir, name).parent.mkdir(exist_ok=True)
    write_csv(rows_path(history_dir, name), rows)
    history["versions"].append({"name": name, "folder": str(folder), "mask_dir": str(mask_dir) if mask_dir else None, "added": datetime.now().isoformat(timespec="seconds"), "n_cases": len(rows)})
    save_history(history_dir, history)
    return f"{name}: {len(rows)} cortes avaliados"


def remove_version(history_dir: Path, name: str) -> None:
    history = load_history(history_dir)
    if not history or not any(version["name"] == name for version in history["versions"]):
        raise PlacarError(f"versão {name!r} não está no histórico")
    # as réguas continuam as da referência original, mesmo que ela seja removida
    history["versions"] = [version for version in history["versions"] if version["name"] != name]
    rows_path(history_dir, name).unlink(missing_ok=True)
    save_history(history_dir, history)


# ---------------------------------------------------------------- placar

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
    for key, *_ in INDICATORS:
        out[key] = bootstrap(np.array([float(row[key]) for row in rows]), np.median, rng)
    for key, _, _, test in RATES:
        out[key] = tuple(100 * value for value in bootstrap(np.array([test(row) for row in rows], dtype=float), np.mean, rng))
    return out


def compare(previous: list[dict], current: list[dict], rng) -> dict:
    """Mudança corte a corte; positivo = a versão nova está mais perto do ideal."""
    out = {}
    for key, _, ideal, _ in INDICATORS:
        gain = np.array([abs(float(a[key]) - ideal) - abs(float(b[key]) - ideal) for a, b in zip(previous, current)])
        out[key] = bootstrap(gain, np.median, rng)
    for key, _, _, test in RATES:
        sign = 1 if key == "aprovada" else -1  # mais aprovadas é melhor; mais brilho sinalizado é pior
        gain = np.array([sign * (float(test(b)) - float(test(a))) for a, b in zip(previous, current)])
        out[key] = tuple(100 * value for value in bootstrap(gain, np.mean, rng))
    return out


def verdict_label(row: dict) -> str:
    """Veredito final com o motivo principal: brilho rejeita sozinho; em revisão, mostra o tipo de defeito."""
    if row["final_status"] == "rejeitar":
        return "rejeitar: brilho"
    if row["final_status"] == "revisar":
        return "revisar: só brilho" if row["wavelet_verdict"] == "textura preservada" else f"revisar: {row['wavelet_verdict']}"
    return "aprovada"


def build_board(history_dir: Path) -> dict | None:
    """Placar de todas as versões do histórico, nos cortes comuns a todas."""
    history = load_history(history_dir)
    if not history or not history["versions"]:
        return None
    names = [version["name"] for version in history["versions"]]
    all_rows = {name: {row["case_id"]: row for row in load_rows(history_dir, name)} for name in names}
    common = sorted(set.intersection(*(set(rows) for rows in all_rows.values())))
    if not common:
        raise PlacarError("nenhum corte em comum entre as versões do histórico")
    rows = {name: [all_rows[name][case] for case in common] for name in names}
    rng = np.random.default_rng(0)
    return {
        "history": history, "names": names, "n_cases": len(common), "rows": rows,
        "dropped": sorted(set.union(*(set(r) for r in all_rows.values())) - set(common)),
        "scores": {name: score(rows[name], rng) for name in names},
        "comparisons": {(a, b): compare(rows[a], rows[b], rng) for a, b in zip(names, names[1:])},
        "verdicts": {name: Counter(verdict_label(row) for row in rows[name]) for name in names},
    }


# ---------------------------------------------------------------- saída

def fmt(value: float, key: str) -> str:
    if key == "wavelet_ratio":
        return f"{value:.2f}"
    if key == "abs_delta_mu":
        return f"{value:.3f}"
    return f"{value:.0f}%"


def fmt_gain(gain: float, key: str) -> str:
    # razão e |Δμ| na própria unidade; porcentagens e taxas em pontos percentuais
    return f"{gain:+.2f}" if key in ("wavelet_ratio", "abs_delta_mu") else f"{gain:+.1f} pp"


def all_indicators():
    """(chave, nome, ideal, unidade) dos indicadores e das taxas, na ordem do placar."""
    return [(key, label, ideal, unit) for key, label, ideal, unit in INDICATORS] + [(key, label, ideal, "% dos cortes") for key, label, ideal, _ in RATES]


def score_table(board: dict) -> list[list[str]]:
    """Linhas [indicador, ideal, versão 1, versão 2, ...] com mediana e intervalo."""
    table = []
    for key, label, ideal, unit in all_indicators():
        cells = [f"{fmt(v, key)} [{fmt(lo, key)}, {fmt(hi, key)}]" for v, lo, hi in (board["scores"][name][key] for name in board["names"])]
        table.append([f"{label} ({unit})", fmt(ideal, key)] + cells)
    return table


def change_table(board: dict) -> list[list[str]]:
    table = []
    for key, label, *_ in all_indicators():
        cells = []
        for pair in board["comparisons"]:
            gain, low, high = board["comparisons"][pair][key]
            cells.append(f"{change_label(gain, low, high, key)} ({fmt_gain(gain, key)})")
        table.append([label] + cells)
    return table


def verdict_table(board: dict) -> list[list[str]]:
    verdicts = board["verdicts"]
    labels = sorted({label for counts in verdicts.values() for label in counts}, key=lambda label: -sum(counts.get(label, 0) for counts in verdicts.values()))
    return [[label] + [str(verdicts[name].get(label, 0)) for name in board["names"]] for label in labels]


def markdown_table(header: list[str], rows: list[list[str]]) -> list[str]:
    def line(cells):
        # a barra de |Δμ| quebraria a coluna
        return "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"
    return [line(header), "|" + "---|" * len(header)] + [line(row) for row in rows]


def write_markdown(board: dict, output: Path) -> None:
    names = board["names"]
    lines = [
        "# Placar por versão", "",
        f"{board['n_cases']} cortes avaliados em todas as versões. Réguas calibradas nas originais de `{board['history']['reference']}`.",
        "Valor = mediana entre os cortes; entre colchetes, intervalo de 95% (bootstrap).", "",
        "## Indicadores", "", *markdown_table(["Indicador", "Ideal", *names], score_table(board)),
        "", "## Vereditos", "", *markdown_table(["Veredito", *names], verdict_table(board)),
    ]
    if board["comparisons"]:
        lines += [
            "", "## Comparação com a versão anterior", "",
            "Ganho = quanto, corte a corte, a versão nova chegou mais perto do ideal (mediana; em taxas, pontos percentuais).",
            "Só é \"melhorou\" ou \"piorou\" quando o intervalo de 95% não cruza o zero **e** a mudança passa do mínimo relevante",
            "(5 pp nas porcentagens e taxas, 0,05 na razão e no |Δμ|); se só a primeira condição vale, aparece \"mudança pequena\".", "",
            *markdown_table(["Indicador", *(f"{a} → {b}" for a, b in board["comparisons"])], change_table(board)),
        ]
    lines += ["", "## Versões", "", *markdown_table(["Versão", "Pasta", "Adicionada em", "Cortes válidos"], [
        [v["name"], f"`{v['folder']}`", v["added"], str(v["n_cases"])] for v in board["history"]["versions"]
    ])]
    if board["dropped"]:
        lines += ["", f"{len(board['dropped'])} cortes ficaram fora do placar por não serem válidos em todas as versões."]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def figure(board: dict):
    """Um painel por indicador ao longo das versões, em preto e branco."""
    names = board["names"]
    panels = all_indicators()
    columns = 4
    rows = -(-len(panels) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 2.9 * rows), squeeze=False)
    axes = axes.ravel()
    for axis in axes[len(panels):]:
        axis.axis("off")
    x = np.arange(len(names))
    for axis, (key, label, ideal, unit) in zip(axes, panels):
        values = np.array([board["scores"][name][key] for name in names])
        axis.errorbar(x, values[:, 0], yerr=[values[:, 0] - values[:, 1], values[:, 2] - values[:, 0]], fmt="o-", color="black", capsize=3, linewidth=1, markersize=4)
        axis.axhline(ideal, color="black", linestyle=":", linewidth=.8)
        axis.set_xticks(x, names, fontsize=8, rotation=30 if len(names) > 3 else 0)
        axis.set_title(f"{label}\n{unit}", fontsize=8)
        axis.tick_params(labelsize=8)
        axis.set_xlim(-.4, len(names) - .6)
    fig.tight_layout()
    return fig


def write_outputs(history_dir: Path, board: dict) -> None:
    names = board["names"]
    write_csv(history_dir / "por_caso.csv", [{"version": name, **row} for name in names for row in board["rows"][name]])
    write_csv(history_dir / "placar.csv", [
        {"version": name, "indicator": key, "value": v, "ci_low": lo, "ci_high": hi}
        for name in names for key, (v, lo, hi) in board["scores"][name].items()
    ])
    write_csv(history_dir / "comparacao.csv", [
        {"from": a, "to": b, "indicator": key, "gain": g, "ci_low": lo, "ci_high": hi, "result": change_label(g, lo, hi, key)}
        for (a, b), values in board["comparisons"].items() for key, (g, lo, hi) in values.items()
    ])
    write_markdown(board, history_dir / "placar.md")
    fig = figure(board)
    fig.savefig(history_dir / "placar.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- linha de comando

def parse_version(text: str) -> tuple[str, Path]:
    name, _, folder = text.partition("=")
    if not folder:
        raise argparse.ArgumentTypeError("use NOME=PASTA, por exemplo v1=CAMINHO/DA/VERSAO1")
    return name, Path(folder)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, required=True, help="pasta do histórico (criada na primeira versão)")
    p.add_argument("--version", type=parse_version, action="append", default=[], help="NOME=PASTA, em ordem cronológica; repita para cada versão nova")
    p.add_argument("--mask-dir", type=Path, help="pasta das máscaras, se não estiverem junto das imagens (guardada no histórico)")
    p.add_argument("--remove", action="append", default=[], help="tira uma versão do histórico")
    a = p.parse_args()
    try:
        for name in a.remove:
            remove_version(a.output_dir, name)
            print(f"{name}: removida")
        for name, folder in a.version:
            print(add_version(a.output_dir, name, folder, a.mask_dir))
        board = build_board(a.output_dir)
    except PlacarError as error:
        raise SystemExit(str(error))
    if board is None:
        raise SystemExit(f"histórico vazio em {a.output_dir}; adicione uma versão com --version NOME=PASTA")
    write_outputs(a.output_dir, board)
    print((a.output_dir / "placar.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
