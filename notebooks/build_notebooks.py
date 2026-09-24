"""Converte os fontes `NN_nome.py` (formato percent) nos `.ipynb` correspondentes.

Os notebooks são gerados, não editados à mão: o fonte `.py` é o que entra no git
(diff legível, sem saídas embutidas) e o `.ipynb` é o artefato de execução.

    python notebooks/build_notebooks.py            # gera todos
    python notebooks/build_notebooks.py 00 01      # gera só esses
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

HERE = Path(__file__).parent


def split_cells(source: str) -> list[tuple[str, str]]:
    """Divide o fonte em (tipo, conteúdo) pelos marcadores `# %%` e `# %% [markdown]`."""
    cells, kind, buf = [], None, []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("# %%"):
            if kind is not None:
                cells.append((kind, "\n".join(buf).strip("\n")))
            kind = "markdown" if "[markdown]" in stripped else "code"
            buf = []
        elif kind is not None:
            buf.append(line[2:] if kind == "markdown" and line.startswith("# ") else
                       ("" if kind == "markdown" and stripped == "#" else line))
    if kind is not None:
        cells.append((kind, "\n".join(buf).strip("\n")))
    return [(k, c) for k, c in cells if c]


def build(path: Path) -> Path:
    text = path.read_text(encoding="utf-8")
    # o docstring do módulo carrega o markdown de abertura
    start = text.index('"""')
    end = text.index('"""', start + 3)
    header, body = text[start + 3:end], text[end + 3:]
    nb = nbformat.v4.new_notebook()
    for kind, content in split_cells(header) + split_cells(body):
        nb.cells.append(nbformat.v4.new_markdown_cell(content) if kind == "markdown"
                        else nbformat.v4.new_code_cell(content))
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    out = path.with_suffix(".ipynb")
    nbformat.write(nb, out)
    return out


if __name__ == "__main__":
    wanted = sys.argv[1:]
    for src in sorted(HERE.glob("[0-9][0-9]_*.py")):
        if wanted and not any(src.name.startswith(w) for w in wanted):
            continue
        out = build(src)
        print(f"{src.name} -> {out.name} ({len(nbformat.read(out, as_version=4).cells)} células)")
