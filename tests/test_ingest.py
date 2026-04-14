"""Testes do ingest."""
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from data.ingest import main


def test_ingest_success(monkeypatch, tmp_path, sample_stock_data):
    def fake_download(*args, **kwargs):
        return sample_stock_data

    monkeypatch.setattr("data.ingest.yf.download", fake_download)

    output = tmp_path / "data.csv"

    args = Namespace(
        ticker="TEST",
        start="2025-04-01",
        end="2025-05-01",
        output=str(output),
    )

    main(args)

    assert output.exists()
    df = pd.read_csv(output)
    assert "Close" in df.columns


def test_ingest_empty(monkeypatch, tmp_path):
    def fake_download(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr("data.ingest.yf.download", fake_download)

    args = Namespace(
        ticker="TEST",
        start="2025-04-01",
        end="2025-05-01",
        output=str(tmp_path / "data.csv"),
    )

    with pytest.raises(ValueError):
        main(args)