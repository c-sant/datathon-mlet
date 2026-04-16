"""Testes do baseline."""

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

from src.models.baseline import (
    avaliar_modelo,
    carregar_dados_csv,
    carregar_pytorch,
    treinar_pytorch,
)


def test_carregar_dados_ok(sample_stock_csv):
    df = carregar_dados_csv(str(sample_stock_csv))
    assert "Close" in df.columns


def test_carregar_dados_sem_close(sample_stock_no_close):
    with pytest.raises(ValueError):
        carregar_dados_csv(str(sample_stock_no_close))


def test_avaliar_modelo():
    scaler = MinMaxScaler()
    scaler.fit([[10], [20], [30]])

    y_real = np.array([0.2, 0.3])
    y_pred = np.array([[0.2], [0.31]])

    result = avaliar_modelo(y_real, y_pred, "Teste", scaler)

    assert "MAE" in result
    assert result["MAE"] >= 0


def test_pytorch_train_and_predict(tmp_path):
    X = np.random.rand(10, 5)
    y = np.random.rand(10)

    model_path = tmp_path / "model.pth"

    treinar_pytorch(X, y, str(model_path))
    preds = carregar_pytorch(X, str(model_path))

    assert preds.shape[0] == len(X)
