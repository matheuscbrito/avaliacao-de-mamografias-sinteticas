"""Bancada de métricas de textura para pares real/sintético do gen-fid."""
from . import io, metrics, spectrum, stats, store          # noqa: F401
from .store import Config                                  # noqa: F401

__all__ = ["io", "metrics", "spectrum", "stats", "store", "Config"]
