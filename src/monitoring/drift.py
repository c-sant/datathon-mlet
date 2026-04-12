"""Detecção de drift para dados de ações da B3 com Evidently.

Monitora distribuição de features financeiras (preço, volume, retorno)
e respostas do pipeline RAG ao longo do tempo.

Thresholds PSI:
  PSI < 0.10  → estável (sem ação)
  PSI ≥ 0.10  → warning (monitorar)
  PSI ≥ 0.20  → crítico (trigger de retrain)
"""
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mlflow
import pandas as pd
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import ColumnDriftMetric
from evidently.report import Report

logger = logging.getLogger(__name__)

DRIFT_REPORT_PATH = Path("data/drift_reports")
DRIFT_LOG_PATH = Path("data/drift_reports/drift_history.json")

PSI_WARNING = 0.10
PSI_CRITICAL = 0.20


@dataclass
class DriftResult:
    """Resultado da análise de drift."""
    timestamp: str
    dataset_drift_detected: bool
    share_drifted_columns: float
    psi_scores: dict[str, float]
    status: str  # "stable", "warning", "critical"
    trigger_retrain: bool
    drifted_features: list[str]


FINANCIAL_FEATURES = [
    "close_price",
    "volume",
    "daily_return",
    "volatility_20d",
    "moving_avg_20d",
    "high_low_range",
    "price_to_ma_ratio",
]


def compute_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features financeiras derivadas para monitoramento de drift.

    Args:
        df: DataFrame com colunas OHLCV (open, high, low, close, volume).

    Returns:
        DataFrame com features financeiras calculadas.
    """
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no DataFrame: {missing}")

    features = pd.DataFrame()
    features["close_price"] = df["close"]
    features["volume"] = df["volume"]
    features["daily_return"] = df["close"].pct_change().fillna(0)
    features["volatility_20d"] = features["daily_return"].rolling(20).std().fillna(0)
    features["moving_avg_20d"] = df["close"].rolling(20).mean().fillna(df["close"])
    features["high_low_range"] = (df["high"] - df["low"]) / df["close"]
    features["price_to_ma_ratio"] = df["close"] / features["moving_avg_20d"]

    return features


def run_drift_detection(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    ticker: str = "B3_ASSET",
    log_to_mlflow: bool = True,
    save_report: bool = True,
) -> DriftResult:
    """Executa detecção de drift em dados financeiros da B3.

    Compara distribuição de features do período de referência (treino)
    com o período atual (produção) usando Evidently PSI.

    Args:
        reference_df: DataFrame de referência (período de treino/baseline).
                      Deve conter colunas OHLCV.
        current_df: DataFrame atual (produção recente).
        ticker: Código do ativo monitorado (ex: "PETR4", "IBOV").
        log_to_mlflow: Se True, loga métricas de drift no MLflow.
        save_report: Se True, salva relatório HTML em disco.

    Returns:
        DriftResult com status e scores PSI por feature.
    """
    logger.info("Executando drift detection para %s...", ticker)
    logger.info(
        "Referência: %d amostras | Atual: %d amostras",
        len(reference_df), len(current_df),
    )

    ref_features = compute_financial_features(reference_df)
    cur_features = compute_financial_features(current_df)

    # Relatório principal: DataDrift + TargetDrift
    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
        *[ColumnDriftMetric(column_name=col) for col in FINANCIAL_FEATURES],
    ])
    report.run(reference_data=ref_features, current_data=cur_features)

    result_dict = report.as_dict()

    # Extrair métricas de drift
    drift_data = result_dict["metrics"][0]["result"]
    dataset_drift = drift_data.get("dataset_drift", False)
    share_drifted = drift_data.get("share_of_drifted_columns", 0.0)

    # PSI por feature
    psi_scores = {}
    drifted_features = []

    for metric in result_dict["metrics"]:
        if metric.get("metric") == "ColumnDriftMetric":
            col = metric["result"]["column_name"]
            psi = metric["result"].get("drift_score", 0.0)
            psi_scores[col] = round(psi, 4)

            if psi >= PSI_WARNING:
                drifted_features.append(col)
                logger.warning(
                    "Drift detectado em %s: PSI=%.4f (%s)",
                    col, psi,
                    "CRÍTICO" if psi >= PSI_CRITICAL else "WARNING",
                )

    # Status global baseado no PSI máximo
    max_psi = max(psi_scores.values()) if psi_scores else 0.0
    if max_psi >= PSI_CRITICAL:
        status = "critical"
        trigger_retrain = True
    elif max_psi >= PSI_WARNING:
        status = "warning"
        trigger_retrain = False
    else:
        status = "stable"
        trigger_retrain = False

    timestamp = datetime.utcnow().isoformat()
    drift_result = DriftResult(
        timestamp=timestamp,
        dataset_drift_detected=dataset_drift,
        share_drifted_columns=round(share_drifted, 4),
        psi_scores=psi_scores,
        status=status,
        trigger_retrain=trigger_retrain,
        drifted_features=drifted_features,
    )

    logger.info(
        "Drift status: %s | PSI máximo: %.4f | Features com drift: %d/%d",
        status.upper(), max_psi, len(drifted_features), len(FINANCIAL_FEATURES),
    )

    if trigger_retrain:
        logger.critical(
            "TRIGGER DE RETRAIN ATIVADO para %s! PSI=%.4f > %.2f",
            ticker, max_psi, PSI_CRITICAL,
        )

    if save_report:
        DRIFT_REPORT_PATH.mkdir(parents=True, exist_ok=True)
        report_path = DRIFT_REPORT_PATH / f"drift_{ticker}_{timestamp[:10]}.html"
        report.save_html(str(report_path))
        logger.info("Relatório de drift salvo: %s", report_path)

        _append_drift_history(drift_result, ticker)

    if log_to_mlflow:
        _log_drift_to_mlflow(drift_result, ticker)

    return drift_result


def _log_drift_to_mlflow(result: DriftResult, ticker: str) -> None:
    """Loga métricas de drift no MLflow para rastreabilidade."""
    try:
        with mlflow.start_run(run_name=f"drift_monitoring_{ticker}"):
            mlflow.log_param("ticker", ticker)
            mlflow.log_param("timestamp", result.timestamp)
            mlflow.log_param("status", result.status)
            mlflow.log_param("trigger_retrain", result.trigger_retrain)

            mlflow.log_metric("share_drifted_columns", result.share_drifted_columns)
            mlflow.log_metric("dataset_drift_detected", int(result.dataset_drift_detected))

            for feature, psi in result.psi_scores.items():
                mlflow.log_metric(f"psi_{feature}", psi)

            mlflow.log_metric("psi_threshold_warning", PSI_WARNING)
            mlflow.log_metric("psi_threshold_critical", PSI_CRITICAL)

        logger.info("Métricas de drift registradas no MLflow")
    except Exception as e:
        logger.warning("Falha ao registrar drift no MLflow: %s", e)


def _append_drift_history(result: DriftResult, ticker: str) -> None:
    """Mantém histórico de drift em arquivo JSON."""
    DRIFT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    history = []
    if DRIFT_LOG_PATH.exists():
        with open(DRIFT_LOG_PATH, encoding="utf-8") as f:
            history = json.load(f)

    history.append({
        "ticker": ticker,
        "timestamp": result.timestamp,
        "status": result.status,
        "trigger_retrain": result.trigger_retrain,
        "share_drifted_columns": result.share_drifted_columns,
        "psi_scores": result.psi_scores,
        "drifted_features": result.drifted_features,
    })

    with open(DRIFT_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def create_synthetic_financial_data(
    n_samples: int = 252,
    drift_factor: float = 0.0,
    ticker: str = "PETR4",
    seed: int = 42,
) -> pd.DataFrame:
    """Gera dados financeiros sintéticos para testes de drift.

    Args:
        n_samples: Número de pregões simulados (252 = 1 ano útil).
        drift_factor: Fator de drift aplicado (0.0 = sem drift, 1.0 = drift severo).
        ticker: Nome do ativo simulado.
        seed: Semente para reprodutibilidade.

    Returns:
        DataFrame com colunas OHLCV sintéticas.
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    base_price = 37.50 if "PETR" in ticker else 68.00
    daily_returns = rng.normal(
        loc=0.0003 + drift_factor * 0.002,
        scale=0.018 + drift_factor * 0.01,
        size=n_samples,
    )
    close = base_price * (1 + daily_returns).cumprod()
    high = close * (1 + rng.uniform(0.002, 0.015, n_samples))
    low = close * (1 - rng.uniform(0.002, 0.015, n_samples))
    open_ = close * (1 + rng.normal(0, 0.005, n_samples))
    volume = rng.integers(
        int(500_000 * (1 + drift_factor)),
        int(5_000_000 * (1 + drift_factor)),
        n_samples,
    ).astype(float)

    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_samples)
    return pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "ticker": ticker,
    })


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Dados de referência (treino — sem drift)
    ref_df = create_synthetic_financial_data(n_samples=252, drift_factor=0.0, ticker="PETR4")

    # Dados atuais (produção — com drift simulado)
    cur_df = create_synthetic_financial_data(n_samples=60, drift_factor=0.4, ticker="PETR4")

    result = run_drift_detection(
        reference_df=ref_df,
        current_df=cur_df,
        ticker="PETR4",
        log_to_mlflow=False,
    )

    print(f"\n=== Drift Detection — PETR4 ===")
    print(f"Status: {result.status.upper()}")
    print(f"Trigger retrain: {result.trigger_retrain}")
    print(f"Features com drift: {result.drifted_features}")
    print(f"\nPSI por feature:")
    for feat, psi in result.psi_scores.items():
        flag = " ⚠" if psi >= PSI_WARNING else ""
        print(f"  {feat:<25} PSI={psi:.4f}{flag}")