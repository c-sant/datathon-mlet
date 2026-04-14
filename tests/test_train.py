"""Testes do train."""
import numpy as np
import pytest

from src.models.train import preparar_series, avaliar_regressao


def test_preparar_series():
    data = np.array([[10], [11], [12], [13], [14]])

    X, y, scaler = preparar_series(data, janela_dias=2)

    assert X.shape[0] > 0
    assert len(y) > 0


def test_preparar_series_erro():
    data = np.array([[10], [11]])

    with pytest.raises(ValueError):
        preparar_series(data, janela_dias=5)


def test_avaliar_regressao():
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaler.fit([[10], [20], [30]])

    y_true = np.array([0.2, 0.3])
    y_pred = np.array([[0.21], [0.31]])

    result = avaliar_regressao(y_true, y_pred, scaler)

    assert "mae" in result
    assert result["mae"] >= 0