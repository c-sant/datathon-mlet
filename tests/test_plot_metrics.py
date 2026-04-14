"""Testes de plot."""
import json
from pathlib import Path

import pytest

from src.utils import plot_metrics


def test_load_metrics(tmp_path, monkeypatch):
    metrics = {
        "baseline": {"mae": 0.4, "rmse": 0.6, "mape": 1.1}
    }

    path = tmp_path / "metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f)

    monkeypatch.setattr(plot_metrics, "METRICS_PATH", str(path))

    df = plot_metrics.load_metrics()

    assert "Modelo" in df.columns


def test_main_creates_files(tmp_path, monkeypatch):
    metrics = {
        "baseline": {"mae": 0.4, "rmse": 0.6, "mape": 1.1}
    }

    metrics_path = tmp_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    monkeypatch.setattr(plot_metrics, "METRICS_PATH", str(metrics_path))
    monkeypatch.setattr(plot_metrics, "OUTPUT_DIR", str(tmp_path))

    plot_metrics.main()

    assert (tmp_path / "comparacao_modelos.png").exists()
    assert (tmp_path / "ranking_mae.png").exists()
    assert (tmp_path / "mape.png").exists()