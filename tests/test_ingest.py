"""Testes do ingest."""

from pytest import raises

from src.models.train import carregar_dados_csv


def test_carregar_dados_csv_success(sample_stock_csv):
    df = carregar_dados_csv(str(sample_stock_csv))

    assert not df.empty
    assert "Close" in df.columns


def test_carregar_dados_csv_sem_close(sample_stock_no_close):
    with raises(ValueError):
        carregar_dados_csv(str(sample_stock_no_close))


def test_carregar_dados_csv_arquivo_inexistente(tmp_path):
    inexistente = tmp_path / "arquivo_inexistente.csv"

    with raises(FileNotFoundError):
        carregar_dados_csv(str(inexistente))
