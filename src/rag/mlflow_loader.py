"""
Loader que transforma resultados do MLflow e relatórios DVC em documentos
para ingestão no FAISS/RAG.

Fontes consumidas:
- MLflow tracking (sqlite:///mlflow/mlflow.db) — runs com params e métricas
- reports/metrics.json          — métricas consolidadas por modelo
- reports/selected_features.csv — features selecionadas pela engenharia
- reports/pca_explained_variance.csv — variância explicada por componente PCA
- params.yaml                   — hiperparâmetros usados no treino
"""
from __future__ import annotations

import json
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Caminhos relativos à raiz do projeto (dentro do container: /app)
# Como o pacote é instalado via pip, __file__ aponta para site-packages.
# Usa /app quando disponível (container); caso contrário, tenta resolver via cwd.
_APP_DIR = Path("/app")
_ROOT = _APP_DIR if _APP_DIR.is_dir() else Path.cwd()
_REPORTS_DIR = _ROOT / "reports"
_PARAMS_FILE = _ROOT / "params.yaml"
_MLFLOW_DB = _ROOT / "mlflow" / "mlflow.db"
_MLFLOW_TRACKING_URI = f"sqlite:///{_MLFLOW_DB}"


# ---------------------------------------------------------------------------
# Helpers de leitura
# ---------------------------------------------------------------------------

def _read_metrics_json() -> dict | None:
    path = _REPORTS_DIR / "metrics.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Não foi possível ler metrics.json: %s", exc)
        return None


def _read_selected_features() -> list[str]:
    path = _REPORTS_DIR / "selected_features.csv"
    if not path.exists():
        return []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [row.get("selected_feature", "").strip() for row in reader if row.get("selected_feature")]
    except Exception as exc:
        logger.warning("Não foi possível ler selected_features.csv: %s", exc)
        return []


def _read_pca_variance() -> list[dict]:
    path = _REPORTS_DIR / "pca_explained_variance.csv"
    if not path.exists():
        return []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception as exc:
        logger.warning("Não foi possível ler pca_explained_variance.csv: %s", exc)
        return []


def _read_params_yaml() -> dict:
    if not _PARAMS_FILE.exists():
        return {}
    try:
        # Evita dependência de pyyaml; faz parsing simples linha a linha
        import yaml  # type: ignore
        with open(_PARAMS_FILE, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("Não foi possível ler params.yaml: %s", exc)
        return {}


def _read_mlflow_runs(max_runs: int = 5) -> list[dict]:
    """Retorna os últimos runs do MLflow com params e métricas."""
    if not _MLFLOW_DB.exists():
        logger.info("mlflow.db não encontrado em %s; pulando runs.", _MLFLOW_DB)
        return []
    try:
        import mlflow  # type: ignore
        mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        runs = []
        for exp in experiments:
            exp_runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                order_by=["start_time DESC"],
                max_results=max_runs,
            )
            runs.extend(exp_runs)
        filtered_runs = []
        for run in runs:
            tags = run.data.tags or {}
            params = run.data.params or {}
            run_name = (tags.get("mlflow.runName") or "").strip().lower()
            source = (tags.get("source") or "").strip().lower()
            stage = (params.get("stage") or "").strip().lower()

            # Ignora runs operacionais do RAG para não poluir o índice.
            if "rag_ingest" in run_name or "rag" in source or "rag_ingest" in stage:
                continue

            filtered_runs.append(run)

        return filtered_runs[:max_runs]
    except Exception as exc:
        logger.warning("Falha ao ler MLflow runs: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Builders de documentos
# ---------------------------------------------------------------------------

def _doc_from_mlflow_run(run, idx: int) -> dict:
    info = run.info
    params = run.data.params or {}
    metrics = run.data.metrics or {}
    tags = run.data.tags or {}

    run_name = tags.get("mlflow.runName", info.run_id[:8])
    ticker = params.get("ticker", "N/A")
    janela = params.get("janela", "N/A")
    epochs = params.get("epochs", "N/A")
    batch = params.get("batch", "N/A")

    # Formata métricas disponíveis
    metric_lines = []
    for key in sorted(metrics):
        metric_lines.append(f"  {key}: {metrics[key]:.4f}")

    metrics_text = "\n".join(metric_lines) if metric_lines else "  (sem métricas registradas)"

    started = ""
    if info.start_time:
        started = datetime.fromtimestamp(info.start_time / 1000, tz=timezone.utc).isoformat()

    text = (
        f"Run MLflow: {run_name}\n"
        f"Ticker: {ticker} | Janela: {janela} dias | Epochs: {epochs} | Batch: {batch}\n"
        f"Status: {info.status}\n"
        f"Métricas:\n{metrics_text}"
    )

    return {
        "id": f"mlflow_run_{idx}_{info.run_id[:8]}",
        "title": f"MLflow Run — {run_name} ({ticker})",
        "text": text,
        "fetched_at": started or datetime.now(timezone.utc).isoformat(),
    }


def _doc_from_metrics_json(metrics: dict) -> dict:
    lines = ["Comparação de modelos treinados (reports/metrics.json):"]
    for model_name, vals in metrics.items():
        mae = vals.get("mae", "N/A")
        rmse = vals.get("rmse", "N/A")
        mape = vals.get("mape", "N/A")
        lines.append(f"  {model_name}: MAE={mae}, RMSE={rmse}, MAPE={mape}%")

    # Melhor modelo por MAE
    try:
        best = min(metrics.items(), key=lambda kv: float(kv[1].get("mae", 9999)))
        lines.append(f"Melhor modelo por MAE: {best[0]} (MAE={best[1].get('mae')})")
    except Exception:
        pass

    return {
        "id": "mlflow_metrics_json",
        "title": "Métricas dos Modelos Treinados",
        "text": "\n".join(lines),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def _doc_from_features(features: list[str], pca_rows: list[dict]) -> dict:
    lines = [f"Features selecionadas pela engenharia de dados: {', '.join(features) if features else 'N/A'}"]

    if pca_rows:
        lines.append("\nVariância explicada por componente PCA:")
        for row in pca_rows:
            comp = row.get("component", "")
            ev = row.get("explained_variance_ratio", "")
            cum = row.get("cumulative_variance", "")
            try:
                lines.append(f"  {comp}: {float(ev)*100:.1f}% (acumulado: {float(cum)*100:.1f}%)")
            except (ValueError, TypeError):
                lines.append(f"  {comp}: {ev} (acumulado: {cum})")

    return {
        "id": "mlflow_features_pca",
        "title": "Features e PCA — Engenharia de Dados",
        "text": "\n".join(lines),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def _doc_from_params(params: dict) -> dict:
    train = params.get("train", {})
    data = params.get("data", {})

    lines = [
        "Parâmetros de treinamento (params.yaml):",
        f"  Ticker de dados: {data.get('ticker', 'N/A')}",
        f"  Período de dados: {data.get('start', 'N/A')} a {data.get('end', 'N/A')}",
        f"  Ticker de treino: {train.get('ticker', 'N/A')}",
        f"  Janela temporal: {train.get('janela', 'N/A')} dias",
        f"  Epochs: {train.get('epochs', 'N/A')}",
        f"  Batch size: {train.get('batch', 'N/A')}",
    ]

    return {
        "id": "mlflow_params_yaml",
        "title": "Parâmetros de Treinamento do Modelo",
        "text": "\n".join(lines),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Função pública
# ---------------------------------------------------------------------------

def load_mlflow_docs() -> list[dict]:
    """
    Carrega documentos sobre o modelo treinado e pipeline DVC para ingestão no RAG.
    Retorna lista de dicts com as chaves: id, title, text, fetched_at.
    """
    docs: list[dict] = []

    # 1. Runs do MLflow (fonte mais rica, opcional)
    runs = _read_mlflow_runs()
    for i, run in enumerate(runs):
        try:
            docs.append(_doc_from_mlflow_run(run, i))
        except Exception as exc:
            logger.warning("Erro ao processar run MLflow #%d: %s", i, exc)

    # 2. Métricas consolidadas do metrics.json
    metrics = _read_metrics_json()
    if metrics:
        docs.append(_doc_from_metrics_json(metrics))

    # 3. Features + PCA
    features = _read_selected_features()
    pca_rows = _read_pca_variance()
    if features or pca_rows:
        docs.append(_doc_from_features(features, pca_rows))

    # 4. Parâmetros de treino
    params = _read_params_yaml()
    if params:
        docs.append(_doc_from_params(params))

    logger.info("mlflow_loader: %d documentos carregados.", len(docs))
    return docs
