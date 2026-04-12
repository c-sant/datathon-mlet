import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "PrevisaoAcoes"
client = MlflowClient()

# Recupera último run do experimento
experiment = client.get_experiment_by_name("previsao_acoes")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["attributes.start_time desc"],
    max_results=1
)

last_run = runs[0]
run_id = last_run.info.run_id
print(f"📂 Último run: {run_id}")

# Registrar modelos de todos os frameworks
registered_versions = {}

for framework in ["pytorch_model", "sklearn_model", "keras_model"]:
    try:
        result = mlflow.register_model(
            f"runs:/{run_id}/{framework}",
            MODEL_NAME
        )
        registered_versions[framework] = result.version
        print(f"✅ {framework} registrado como versão {result.version}")
    except Exception as e:
        print(f"⚠️ Não foi possível registrar {framework}: {e}")

# Recuperar métricas do último run
metrics = last_run.data.metrics
mae = metrics.get("MAE_pytorch", None)
rmse = metrics.get("RMSE_pytorch", None)

# Critério de promoção (exemplo: RMSE < 0.05)
for framework, version in registered_versions.items():
    if rmse is not None and rmse < 0.05:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage="Production"
        )
        print(f"🚀 {framework} promovido para Production (RMSE={rmse:.4f})")
    else:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version,
            stage="Staging"
        )
        print(f"📌 {framework} mantido em Staging (RMSE={rmse})")
