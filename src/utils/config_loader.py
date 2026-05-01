"""Carrega o model_config.yaml do diretorio config/."""

import os
from pathlib import Path

import yaml

def _resolve_config_path() -> Path:
    """Resolve o caminho do arquivo de configuracao para diferentes ambientes."""
    env_path = os.getenv("MODEL_CONFIG_PATH")
    if env_path:
        return Path(env_path)

    candidates = [
        # Execucao no repositorio com PYTHONPATH=.
        Path(__file__).resolve().parents[2] / "config" / "model_config.yaml",
        # Execucao em container com WORKDIR=/app e codigo copiado no volume.
        Path("/app/config/model_config.yaml"),
        # Fallback para execucao local fora do repositorio.
        Path.cwd() / "config" / "model_config.yaml",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Preserva mensagem de erro clara apontando para o local principal esperado.
    return candidates[0]


_CONFIG_PATH = _resolve_config_path()

_cache: dict | None = None


def load_config() -> dict:
    """Retorna o conteúdo do model_config.yaml (com cache em memória)."""
    global _cache
    if _cache is None:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            _cache = yaml.safe_load(f)
    return _cache
