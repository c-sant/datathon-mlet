"""
benchmark.py
Script de avaliação comparativa dos modelos treinados (PyTorch, Sklearn, Keras).
Local: evaluation/

Funções:
- Recupera runs do MLflow para o experimento 'previsao_acoes'
- Extrai métricas (MAE, RMSE, MAPE) e parâmetros
- Gera tabela comparativa (benchmark.csv)
- Cria gráficos de barras comparando métricas por framework
- Loga tudo como artifacts no MLflow
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

ROOT_DIR = Path(__file__).resolve().parents[1]
TRACKING_URI = f"sqlite:///{(ROOT_DIR / 'mlflow' / 'mlflow.db').resolve().as_posix()}"
FRAMEWORK_PREFIXES = {
    "PyTorch": "pytorch",
    "Sklearn": "sklearn",
    "Keras": "keras",
}


def gerar_benchmark(experiment_name="previsao_acoes"):
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)

    # Recupera o experimento
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experimento {experiment_name} não encontrado.")

    # Busca os últimos runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time desc"],
        max_results=50,
    )

    registros = []
    for run in runs:
        metrics = run.data.metrics
        params = run.data.params

        # Detecta framework dinamicamente
        framework = None
        if "mae_pytorch" in metrics:
            framework = "PyTorch"
        elif "mae_sklearn" in metrics:
            framework = "Sklearn"
        elif "mae_keras" in metrics:
            framework = "Keras"
        elif "framework" in params:
            framework = params["framework"]

        # Só adiciona se framework foi identificado
        if framework:
            metric_prefix = FRAMEWORK_PREFIXES.get(framework, framework.lower())
            registros.append(
                {
                    "Run_ID": run.info.run_id,
                    "Ticker": params.get("ticker", ""),
                    "Framework": framework,
                    "MAE": metrics.get(f"mae_{metric_prefix}"),
                    "RMSE": metrics.get(f"rmse_{metric_prefix}"),
                    "MAPE": metrics.get(f"mape_{metric_prefix}"),
                    "Janela": params.get("janela", ""),
                    "Epochs": params.get("epochs", ""),
                    "Batch": params.get("batch_size", ""),
                }
            )

    # Cria DataFrame
    df = pd.DataFrame(registros)
    if df.empty:
        raise RuntimeError(
            "Nenhum run com métricas compatíveis foi encontrado para gerar o benchmark."
        )

    # Salva como CSV
    df.to_csv("benchmark.csv", index=False)

    # Cria gráficos comparativos
    for metric in ["MAE", "RMSE", "MAPE"]:
        plt.figure(figsize=(6, 4))
        df.groupby("Framework")[metric].mean().plot(
            kind="bar", color=["#1f77b4", "#ff7f0e", "#2ca02c"]
        )
        plt.title(f"Comparação de {metric} por Framework")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(f"benchmark_{metric}.png")
        plt.close()

    # Loga artifacts no MLflow
    with mlflow.start_run(run_name="benchmark_documentado"):
        mlflow.log_artifact("benchmark.csv")
        mlflow.log_artifact("benchmark_MAE.png")
        mlflow.log_artifact("benchmark_RMSE.png")
        mlflow.log_artifact("benchmark_MAPE.png")

    print("✅ Benchmark gerado e logado como artifact no MLflow.")
    print(df)


if __name__ == "__main__":
    gerar_benchmark()
