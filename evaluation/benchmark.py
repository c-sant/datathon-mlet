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

import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient


def gerar_benchmark(experiment_name="previsao_acoes"):
    client = MlflowClient()

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
        if "MAE_pytorch" in metrics:
            framework = "PyTorch"
        elif "MAE_sklearn" in metrics:
            framework = "Sklearn"
        elif "MAE_keras" in metrics:
            framework = "Keras"
        elif "framework" in params:
            framework = params["framework"]

        # Só adiciona se framework foi identificado
        if framework:
            registros.append(
                {
                    "Run_ID": run.info.run_id,
                    "Ticker": params.get("ticker", ""),
                    "Framework": framework,
                    "MAE": metrics.get(f"MAE_{framework.lower()}", None),
                    "RMSE": metrics.get(f"RMSE_{framework.lower()}", None),
                    "MAPE": metrics.get(f"MAPE_{framework.lower()}", None),
                    "Janela": params.get("janela", ""),
                    "Epochs": params.get("epochs", ""),
                    "Batch": params.get("batch_size", ""),
                }
            )

    # Cria DataFrame
    df = pd.DataFrame(registros)

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
