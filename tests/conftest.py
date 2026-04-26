import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

for path in (str(ROOT_DIR), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture
def sample_stock_data():
    dates = pd.date_range("2025-04-01", periods=20, freq="D")
    return pd.DataFrame(
        {
            "Open": np.linspace(10, 20, 20),
            "High": np.linspace(11, 21, 20),
            "Low": np.linspace(9, 19, 20),
            "Close": np.linspace(10.5, 20.5, 20),
            "Volume": np.arange(1000, 1020),
        },
        index=dates,
    )


@pytest.fixture
def sample_stock_csv(tmp_path, sample_stock_data):
    path = tmp_path / "stock.csv"
    sample_stock_data.to_csv(path)
    return path


@pytest.fixture
def sample_stock_no_close(tmp_path):
    df = pd.DataFrame({"Open": [1, 2, 3]})
    path = tmp_path / "no_close.csv"
    df.to_csv(path)
    return path
