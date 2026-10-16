"""Placar por versão do gerador: brilho e textura melhoraram de uma iteração para a outra?

Cada versão é uma pasta com as sintéticas que uma iteração do modelo gerou a
partir das **mesmas originais** (mesmos nomes de arquivo). O placar guarda um
**histórico** numa pasta: cada versão é avaliada uma única vez e versões novas
podem ser adicionadas a qualquer momento.

1. A primeira versão adicionada vira a referência: as réguas são calibradas nas
   originais dela e guardadas no histórico. Toda versão seguinte é medida com
   essas mesmas réguas, e as originais dela precisam ser idênticas às da
   referência — senão os números não se comparam.
2. Cada par é conferido (`regional.py`): o modo da versão (mesma anatomia ou só
   máscara) é detectado e, no modo mesma anatomia, pares com correlação baixa
   com a original são rejeitados como suspeitos de arquivo trocado.
3. O placar usa só os cortes válidos em todas as versões do histórico e separa
   os indicadores em três grupos: veredito final, brilho e textura.
4. Cada indicador sai como mediana entre os cortes (ou % dos cortes), com
   intervalo de 95% por bootstrap.
5. Cada versão é comparada com a anterior, corte a corte: o indicador chegou
   mais perto do ideal? Só diz "melhorou" ou "piorou" quando o intervalo da
   diferença não cruza o zero e a mudança passa do mínimo relevante.

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
from typing import Callable, NamedTuple

import matplotlib.pyplot as plt
import numpy as np

from wavelet.src.load_data import CasePaths, discover_cases
from wavelet.src.run_pipeline import write_csv
from wavelet.src.validate_cases import ValidCase, validate_case
from frameworks.texture_evaluator.regional import apply_pair_check
from frameworks.texture_evaluator.run_mvp import calibrate, context_from_dict, context_to_dict, evaluate_pair

BOOTSTRAP = 2000
HISTORY_FILE = "historico.json"
GROUPS = {"final": "Veredito final", "brilho": "Brilho", "textura": "Textura"}


class Indicator(NamedTuple):
    key: str
    label: str        # a técnica, como aparece na tabela
    meaning: str      # o que ela mede, em linguagem simples
    group: str
    ideal: float
    unit: str
    # menor mudança que conta como real (exploratório): abaixo disso, mesmo que
    # o intervalo não cruze o zero, o placar diz "mudança pequena"
    min_change: float
    # None: mediana da coluna `key`; função: % dos cortes em que ela é verdadeira
    test: Callable[[dict], bool] | None = None
    # figura de visual.py que mostra o que a técnica enxerga (None: sem figura)
    image: str | None = None


# o ideal é a sintética igual à original
INDICATORS = (
    Indicator("aprovada", "Veredito final", "cortes aprovados (sem sinal de brilho nem de textura)", "final", 100.0, "% dos cortes", 5.0,
              test=lambda row: row["final_status"] == "aprovada"),
    Indicator("abs_delta_mu", "Filtro de brilho: Δμ", "diferença da média de todos os pixels do pulmão", "brilho", 0.0, "|sintética − original|", .05,
              image="delta_mu"),
    Indicator("brilho", "Filtro de brilho + densidade", "cortes com brilho fora do aceitável (Δμ, saturação ou pixels fora da faixa real)", "brilho", 0.0, "% dos cortes", 5.0,
              test=lambda row: row["brightness_status"] != "approved" or row["density_status"] != "ok", image="densidade"),
    Indicator("abs_delta_ll", "Wavelet LL (média local, 4×4 px)", "diferença do brilho médio vista pela wavelet, em fração da régua", "brilho", 0.0, "|sintética − original|", .05,
              image="ll"),
    Indicator("nex", "NEx na ROI pulmonar", "excesso de energia perto de Nyquist nas janelas internas do pulmão; fora de P5–P95 real pede revisão", "textura", 0.0, "% dos cortes", 5.0,
              test=lambda row: row.get("nex_status") in {"alto", "baixo"}),
    Indicator("map_area_brilho_local_pct", "Mapa: brilho por bloco (16 px)", "área do pulmão em que o bloco clareou ou escureceu além da tolerância", "brilho", 0.0, "% do pulmão", 5.0),
    Indicator("wavelet_fine_pct", "Wavelet 2 px (textura fina)", "detalhe entre pixels a ~2 px: bordas finas, ruído, pequenos vasos", "textura", 100.0, "% da original", 5.0,
              image="wavelet_fina"),
    Indicator("wavelet_coarse_pct", "Wavelet 4 px (textura grossa)", "detalhe em estruturas de ~4 px: vasos médios, septos", "textura", 100.0, "% da original", 5.0,
              image="wavelet_grossa"),
    Indicator("wavelet_ratio", "Wavelet 2 px ÷ 4 px (tipo de perda)", "abaixo de 1: perdeu mais o fino (borrou); acima de 1: sobrou fino demais (ruído)", "textura", 1.0, "razão", .05,
              image="wavelet_fina"),
    Indicator("glcm_contrast_pct", "GLCM contraste (vizinhos a 1 px)", "quanto o tom muda de um pixel para o vizinho", "textura", 100.0, "% da original", 5.0,
              image="glcm_contraste"),
    Indicator("glcm_energy_pct", "GLCM energia (uniformidade)", "quanto os pares de tons vizinhos se repetem; sobe quando a imagem fica lisa", "textura", 100.0, "% da original", 5.0),
    Indicator("textura", "Veredito Wavelet", "cortes com textura alterada (borrou, ruído, perdeu ou ganhou contraste, estourou)", "textura", 0.0, "% dos cortes", 5.0,
              test=lambda row: row["wavelet_verdict"] != "textura preservada"),
    Indicator("map_area_textura_alterada_pct", "Mapa: textura por bloco (16 px)", "área do pulmão em que o bloco tem textura alterada", "textura", 0.0, "% do pulmão", 5.0),
    Indicator("map_corr_textura", "Mapa: correlação espacial", "a textura está nos mesmos lugares da original? (1 = sim)", "textura", 1.0, "correlação", .05),
)
DECIMALS = {"wavelet_ratio": 2, "map_corr_textura": 2, "abs_delta_mu": 3, "abs_delta_ll": 3}
FILTER_TEXT = {"approved": "filtro: aprovado", "manual_review": "filtro: revisão", "hard_reject": "filtro: rejeição"}
DENSITY_TEXT = {"ok": "densidade: ok", "brighter": "densidade: clareou", "darker": "densidade: escureceu", "too_bright": "densidade: claro demais", "too_dark": "densidade: escuro demais"}


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
        rows = list(csv.DictReader(handle))
    # colunas derivadas, calculadas na leitura para valer também em históricos já gravados
    for row in rows:
        row["abs_delta_ll"] = abs(float(row["wavelet_ll_synthetic"]) - float(row["wavelet_ll_real"]))
        row["glcm_energy_pct"] = 100 * float(row["glcm_energy_synthetic"]) / float(row["glcm_energy_real"])
    return rows


def case_paths(folder: Path, mask_dir: Path | None) -> list[CasePaths]:
    paths = discover_cases(folder)
    if mask_dir:
        paths = [CasePaths(p.case_id, p.original, p.synthetic, mask_dir / f"{p.case_id}_mask.tiff") for p in paths]
    return paths


def load_valid(folder: Path, mask_dir: Path | None) -> dict[str, ValidCase]:
    valid = {}
    for paths in case_paths(folder, mask_dir):
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
    if history and "maps" not in history["calibration"]:
        raise PlacarError(f"o histórico em {history_dir} foi criado por uma versão antiga do placar (sem mapa por região); apague a pasta e adicione as versões de novo")
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
    mode = apply_pair_check(rows)
    suspicious = sum(row["pair_status"] == "suspeito" for row in rows)
    history_dir.mkdir(parents=True, exist_ok=True)
    rows_path(history_dir, name).parent.mkdir(exist_ok=True)
    write_csv(rows_path(history_dir, name), rows)
    history["versions"].append({
        "name": name, "folder": str(folder), "mask_dir": str(mask_dir) if mask_dir else None,
        "added": datetime.now().isoformat(timespec="seconds"), "n_cases": len(rows), "mode": mode, "suspicious_pairs": suspicious,
    })
    save_history(history_dir, history)
    return f"{name}: {len(rows)} cortes avaliados · modo {mode}" + (f" · {suspicious} pares suspeitos" if suspicious else "")


def refresh_history(history_dir: Path) -> str:
    """Recalcula versões já guardadas quando o protocolo ganha uma nova métrica.

    A referência continua sendo a versão original do histórico; só as colunas
    derivadas são refeitas com a calibração atual.
    """
    history = load_history(history_dir)
    if not history or not history["versions"]:
        raise PlacarError("não há versões para recalcular")
    reference = next(version for version in history["versions"] if version["name"] == history["reference"])
    reference_masks = Path(reference["mask_dir"]) if reference.get("mask_dir") else None
    reference_valid = load_valid(Path(reference["folder"]), reference_masks)
    if not reference_valid:
        raise PlacarError("não foi possível recarregar a versão de referência")
    ctx = calibrate(list(reference_valid.values()))
    history["calibration"] = context_to_dict(ctx)
    for version in history["versions"]:
        masks = Path(version["mask_dir"]) if version.get("mask_dir") else None
        valid = load_valid(Path(version["folder"]), masks)
        cases = sorted(case for case in valid if case in history["originals"])
        rows = [{**evaluate_pair(case, valid[case].original, valid[case].synthetic, valid[case].lung_mask, ctx), "abs_delta_mu": 0.0} for case in cases]
        for row in rows:
            row["abs_delta_mu"] = abs(row["delta_mu"])
        mode = apply_pair_check(rows)
        version.update({"n_cases": len(rows), "mode": mode, "suspicious_pairs": sum(row["pair_status"] == "suspeito" for row in rows)})
        write_csv(rows_path(history_dir, version["name"]), rows)
    save_history(history_dir, history)
    return f"{len(history['versions'])} versão(ões) recalculada(s) com a calibração atual"


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
    """Estatística e intervalo de 95% reamostrando os cortes; sem valores, devolve vazio."""
    if len(values) == 0:
        return (float("nan"),) * 3
    samples = [statistic(values[rng.integers(0, len(values), len(values))]) for _ in range(BOOTSTRAP)]
    low, high = np.percentile(samples, (2.5, 97.5))
    return float(statistic(values)), float(low), float(high)


def change_label(gain: float, low: float, high: float, indicator: Indicator) -> str:
    if not np.isfinite(gain):
        return "n/a"
    if low <= 0 <= high:
        return "sem diferença clara"
    if abs(gain) < indicator.min_change:
        return "mudança pequena"
    return "melhorou" if gain > 0 else "piorou"


def trusted(row: dict) -> bool:
    """Par suspeito (arquivo trocado?) só conta no veredito final; brilho e textura dele não valem nada."""
    return row.get("pair_status") != "suspeito"


def values(rows: list[dict], indicator: Indicator) -> np.ndarray:
    if indicator.group != "final":
        rows = [row for row in rows if trusted(row)]
    if indicator.test:
        return np.array([indicator.test(row) for row in rows], dtype=float)
    out = np.array([float(row[indicator.key]) for row in rows])
    # colunas do mapa ficam vazias no modo só máscara
    return out[np.isfinite(out)]


def score(rows: list[dict], rng) -> dict:
    out = {}
    for indicator in INDICATORS:
        if indicator.test:
            out[indicator.key] = tuple(100 * value for value in bootstrap(values(rows, indicator), np.mean, rng))
        else:
            out[indicator.key] = bootstrap(values(rows, indicator), np.median, rng)
    return out


def compare(previous: list[dict], current: list[dict], rng) -> dict:
    """Mudança corte a corte; positivo = a versão nova está mais perto do ideal."""
    out = {}
    every = list(zip(previous, current))
    reliable = [(a, b) for a, b in every if trusted(a) and trusted(b)]
    for indicator in INDICATORS:
        pairs_ = every if indicator.group == "final" else reliable
        if indicator.test:
            sign = 1 if indicator.ideal else -1  # ideal 100%: mais é melhor; ideal 0%: menos é melhor
            gain = np.array([sign * (float(indicator.test(b)) - float(indicator.test(a))) for a, b in pairs_])
            out[indicator.key] = tuple(100 * value for value in bootstrap(gain, np.mean, rng))
            continue
        pairs = [(float(a[indicator.key]), float(b[indicator.key])) for a, b in pairs_]
        gain = np.array([abs(x - indicator.ideal) - abs(y - indicator.ideal) for x, y in pairs if np.isfinite(x) and np.isfinite(y)])
        out[indicator.key] = bootstrap(gain, np.median, rng)
    return out


def verdict_label(row: dict) -> str:
    """Veredito final com o motivo principal: par suspeito e brilho rejeitam; em revisão, mostra o tipo de defeito."""
    if row.get("pair_status") == "suspeito":
        return "rejeitar: par suspeito"
    if row["final_status"] == "rejeitar":
        return "rejeitar: brilho"
    if row["final_status"] == "revisar":
        if row.get("nex_status") in {"alto", "baixo"}:
            return f"revisar: NEx {row['nex_status']}"
        return "revisar: só brilho" if row["wavelet_verdict"] == "textura preservada" else f"revisar: {row['wavelet_verdict']}"
    return "aprovada"


COUNTS = {
    "final": lambda row: [verdict_label(row)],
    "brilho": lambda row: [FILTER_TEXT[row["brightness_status"]], DENSITY_TEXT[row["density_status"]]],
    "textura": lambda row: [row["wavelet_verdict"], f"NEx: {row.get('nex_status', 'indisponível')}"] ,
}


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
        "counts": {group: {name: Counter(label for row in rows[name] if group == "final" or trusted(row) for label in count(row)) for name in names} for group, count in COUNTS.items()},
    }


def load_case(entry: dict, case_id: str) -> ValidCase:
    paths = next((p for p in case_paths(Path(entry["folder"]), Path(entry["mask_dir"]) if entry["mask_dir"] else None) if p.case_id == case_id), None)
    if paths is None:
        raise PlacarError(f"{case_id} não encontrado em {entry['folder']}")
    case = validate_case(paths, erosion_radius=2)
    if not isinstance(case, ValidCase):
        raise PlacarError(f"{case_id} em {entry['name']}: {case.reason}")
    return case


# ---------------------------------------------------------------- saída

def fmt(value: float, indicator: Indicator) -> str:
    if not np.isfinite(value):
        return "n/a"
    if indicator.key in DECIMALS:
        return f"{value:.{DECIMALS[indicator.key]}f}"
    return f"{value:.0f}%"


def fmt_gain(gain: float, indicator: Indicator) -> str:
    # razão, correlação e |Δμ| na própria unidade; porcentagens e taxas em pontos percentuais
    return f"{gain:+.{DECIMALS[indicator.key]}f}" if indicator.key in DECIMALS else f"{gain:+.1f} pp"


def group_indicators(group: str) -> list[Indicator]:
    return [indicator for indicator in INDICATORS if indicator.group == group]


def technique_table(board: dict, group: str) -> tuple[list[str], list[list[str]]]:
    """Uma linha por técnica: o que mede, unidade, ideal, score de cada versão e mudança entre versões."""
    pairs = [f"{a} → {b}" for a, b in board["comparisons"]]
    header = ["Técnica", "O que mede", "Unidade", "Ideal", *board["names"], *pairs]
    rows = []
    for indicator in group_indicators(group):
        cells = [indicator.label, indicator.meaning, indicator.unit, fmt(indicator.ideal, indicator)]
        for name in board["names"]:
            value, low, high = board["scores"][name][indicator.key]
            cells.append(f"{fmt(value, indicator)} [{fmt(low, indicator)}, {fmt(high, indicator)}]" if np.isfinite(value) else "n/a")
        for pair in board["comparisons"]:
            gain, low, high = board["comparisons"][pair][indicator.key]
            label = change_label(gain, low, high, indicator)
            cells.append(label if label == "n/a" else f"{label} ({fmt_gain(gain, indicator)})")
        rows.append(cells)
    return header, rows


def count_table(board: dict, group: str) -> list[list[str]]:
    counts = board["counts"][group]
    labels = sorted({label for c in counts.values() for label in c}, key=lambda label: (label.split(":")[0], -sum(c.get(label, 0) for c in counts.values())))
    return [[label] + [str(counts[name].get(label, 0)) for name in board["names"]] for label in labels]


def version_table(board: dict) -> list[list[str]]:
    return [[v["name"], v["folder"], v.get("mode", "?"), str(v.get("suspicious_pairs", "?")), v["added"].replace("T", " "), str(v["n_cases"])] for v in board["history"]["versions"]]


VERSION_HEADER = ["Versão", "Pasta", "Modo", "Pares suspeitos", "Adicionada em", "Cortes válidos"]
CHANGE_NOTE = (
    "Ganho = quanto, corte a corte, a versão nova chegou mais perto do ideal (mediana; em taxas, pontos percentuais). "
    "\"melhorou\" ou \"piorou\" só quando o intervalo de 95% não cruza o zero e a mudança passa do mínimo relevante "
    "(5 pp nas porcentagens e taxas; 0,05 na razão, na correlação e no |Δμ|); fora do zero mas abaixo do mínimo: \"mudança pequena\". "
    "n/a: o mapa compara cada bloco com o mesmo bloco da original; numa versão só máscara (outro pulmão no mesmo contorno) esse par de blocos não existe."
)


def markdown_table(header: list[str], rows: list[list[str]]) -> list[str]:
    def line(cells):
        # a barra de |Δμ| quebraria a coluna
        return "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"
    return [line(header), "|" + "---|" * len(header)] + [line(row) for row in rows]


def write_markdown(board: dict, output: Path) -> None:
    names = board["names"]
    pairs = [f"{a} → {b}" for a, b in board["comparisons"]]
    lines = [
        "# Placar por versão", "",
        f"{board['n_cases']} cortes avaliados em todas as versões. Réguas calibradas nas originais de `{board['history']['reference']}`.",
        "Valor = mediana entre os cortes (ou % dos cortes); entre colchetes, intervalo de 95% (bootstrap).", "",
        "## Versões", "", *markdown_table(VERSION_HEADER, version_table(board)),
    ]
    for group, title in GROUPS.items():
        lines += ["", f"## {title}", "", *markdown_table(*technique_table(board, group))]
        lines += ["", *markdown_table(["Contagem de cortes", *names], count_table(board, group))]
    if pairs:
        lines += ["", CHANGE_NOTE]
    if board["dropped"]:
        lines += ["", f"{len(board['dropped'])} cortes ficaram fora do placar por não serem válidos em todas as versões."]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def figure(board: dict, group: str | None = None):
    """Um painel por indicador ao longo das versões, em preto e branco."""
    names = board["names"]
    panels = group_indicators(group) if group else list(INDICATORS)
    columns = 4
    rows = -(-len(panels) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 2.9 * rows), squeeze=False)
    axes = axes.ravel()
    for axis in axes[len(panels):]:
        axis.axis("off")
    x = np.arange(len(names))
    for axis, indicator in zip(axes, panels):
        data = np.array([board["scores"][name][indicator.key] for name in names])
        axis.errorbar(x, data[:, 0], yerr=[data[:, 0] - data[:, 1], data[:, 2] - data[:, 0]], fmt="o-", color="black", capsize=3, linewidth=1, markersize=4)
        axis.axhline(indicator.ideal, color="black", linestyle=":", linewidth=.8)
        axis.set_xticks(x, names, fontsize=8, rotation=30 if len(names) > 3 else 0)
        axis.set_title(f"{indicator.label}\n{indicator.unit}", fontsize=8)
        axis.tick_params(labelsize=8)
        axis.set_xlim(-.4, len(names) - .6)
    fig.tight_layout()
    return fig


def write_outputs(history_dir: Path, board: dict) -> None:
    names = board["names"]
    by_key = {indicator.key: indicator for indicator in INDICATORS}
    write_csv(history_dir / "por_caso.csv", [{"version": name, **row} for name in names for row in board["rows"][name]])
    write_csv(history_dir / "placar.csv", [
        {"version": name, "group": by_key[key].group, "indicator": key, "value": v, "ci_low": lo, "ci_high": hi}
        for name in names for key, (v, lo, hi) in board["scores"][name].items()
    ])
    write_csv(history_dir / "comparacao.csv", [
        {"from": a, "to": b, "group": by_key[key].group, "indicator": key, "gain": g, "ci_low": lo, "ci_high": hi, "result": change_label(g, lo, hi, by_key[key])}
        for (a, b), values_ in board["comparisons"].items() for key, (g, lo, hi) in values_.items()
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
