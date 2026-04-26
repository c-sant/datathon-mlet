"""Carrega o model_config.yaml do diretório config/ na raiz do projeto."""

from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "model_config.yaml"

_cache: dict | None = None


def load_config() -> dict:
    """Retorna o conteúdo do model_config.yaml (com cache em memória)."""
    global _cache
    if _cache is None:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            _cache = yaml.safe_load(f)
    return _cache
