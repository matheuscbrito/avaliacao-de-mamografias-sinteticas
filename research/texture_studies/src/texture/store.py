"""Configuração, hash de proveniência e armazenamento em parquet longo.

Extraído da antiga bancada INbreast (`extracao_metricas_textura.ipynb`,
seções 2 e 3; já removida do repo). Regras:

1. `paths` nunca entra em hash. Todo o resto entra, por subárvore.
2. Toda linha gravada carrega `config_hash`. Re-executar com o mesmo hash
   substitui as linhas daquela chave; com hash diferente, coexiste.
3. Nenhuma linha é removida por QC — QC é coluna de flag.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def observed_versions() -> dict:
    import matplotlib, pyarrow, scipy, skimage, tifffile
    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        "numpy": np.__version__, "scipy": scipy.__version__, "scikit-image": skimage.__version__,
        "pandas": pd.__version__, "pyarrow": pyarrow.__version__,
        "matplotlib": matplotlib.__version__, "tifffile": tifffile.__version__,
        "platform": platform.platform(),
    }


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


class Config:
    """`config.yaml` carregado + hash por subárvore."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else ROOT / "config.yaml"
        with open(self.path, encoding="utf-8") as f:
            self.cfg: dict = yaml.safe_load(f)
        self.env = observed_versions()
        self.paths = {k: (ROOT / v if not Path(v).is_absolute() else Path(v))
                      for k, v in self.cfg["paths"].items()}
        self.out = self.paths["out_dir"]
        self.out.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, key: str) -> dict:
        if key not in self.cfg or key == "paths":
            raise KeyError(f"subárvore de configuração inexistente ou proibida: {key!r}")
        return self.cfg[key]

    def hash(self, *subtrees: str) -> str:
        """sha256 (12 hex) de env + common + subárvores. `paths` nunca entra."""
        payload = {"env": self.env, "common": self.cfg["common"]}
        for name in subtrees:
            payload[name] = self[name]
        return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]

    def sidecar(self, name: str, subtrees: Sequence[str], **extra) -> Path:
        """Grava `outputs/<name>.json` com parâmetros, hashes, versões e HEAD."""
        doc = {
            "timestamp": now_iso(), "git_head": git_head(), "env": self.env,
            "common": self.cfg["common"], "subtrees": {s: self[s] for s in subtrees},
            "config_hash": self.hash(*subtrees), **extra,
        }
        p = self.out / f"{name}.json"
        p.write_text(json.dumps(doc, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return p


def read_parquet_or_empty(path: Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def upsert_parquet(path: Path, new: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    """Grava `new` em `path` substituindo linhas com as mesmas `keys`. Nunca duplica."""
    missing = [k for k in keys if k not in new.columns]
    if missing:
        raise KeyError(f"{Path(path).name}: colunas de chave ausentes: {missing}")
    if new.empty:
        return read_parquet_or_empty(path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    old = read_parquet_or_empty(path)
    if not old.empty:
        old_missing = [k for k in keys if k not in old.columns]
        if old_missing:
            raise KeyError(f"{path} existe sem as chaves {old_missing}; schema mudou — renomeie o arquivo.")
        incoming = set(map(tuple, new[list(keys)].astype(str).to_numpy()))
        keep = ~pd.Series(list(map(tuple, old[list(keys)].astype(str).to_numpy())), index=old.index).isin(incoming)
        old = old[keep]
    out = pd.concat([old, new], ignore_index=True) if not old.empty else new.reset_index(drop=True)
    out.to_parquet(path, index=False)
    return out


METRIC_KEYS = ["pair_id", "set", "metric", "config_hash"]
PROFILE_KEYS = ["pair_id", "set", "radial_bin", "config_hash"]
QC_KEYS = ["pair_id", "flag", "config_hash"]


def write_long(path: Path, rows: list[dict], keys: Sequence[str]) -> pd.DataFrame:
    """Grava uma lista de dicts em formato longo, idempotente por `keys`."""
    if not rows:
        return read_parquet_or_empty(path)
    df = pd.DataFrame(rows)
    df["timestamp"] = now_iso()
    return upsert_parquet(path, df, keys)


def metrics_wide(df: pd.DataFrame, config_hash: str | None = None) -> pd.DataFrame:
    """Pivota `metrics.parquet` (longo) para `pair_id × set` nas colunas de métrica."""
    if config_hash is not None:
        df = df[df["config_hash"] == config_hash]
    return df.pivot_table(index=["pair_id", "set"], columns="metric", values="value", aggfunc="last")
