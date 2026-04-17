from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "PrevisaoAcoes"
ROOT_DIR = Path(__file__).resolve().parents[1]
TRACKING_URI = f"sqlite:///{(ROOT_DIR / 'mlflow' / 'mlflow.db').resolve().as_posix()}"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(tracking_uri=TRACKING_URI)

# Recupera último run do experimento
experiment = client.get_experiment_by_name("previsao_acoes")
if experiment is None:
    raise RuntimeError("Experimento 'previsao_acoes' não encontrado no tracking store configurado.")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["attributes.start_time desc"],
    max_results=1,
)

if not runs:
    raise RuntimeError("Nenhum run encontrado para o experimento 'previsao_acoes'.")

last_run = runs[0]
run_id = last_run.info.run_id
print(f"📂 Último run: {run_id}")

# Registrar modelos de todos os frameworks
registered_versions = {}

for framework in ["pytorch_model", "sklearn_model", "keras_model"]:
    try:
        result = mlflow.register_model(f"runs:/{run_id}/{framework}", MODEL_NAME)
        registered_versions[framework] = result.version
        print(f"✅ {framework} registrado como versão {result.version}")
    except Exception as e:
        print(f"⚠️ Não foi possível registrar {framework}: {e}")

# Recuperar métricas do último run
metrics = last_run.data.metrics
mae = metrics.get("mae_pytorch")
rmse = metrics.get("rmse_pytorch")

# Critério de promoção (exemplo: RMSE < 0.05)
for framework, version in registered_versions.items():
    if rmse is not None and rmse < 0.05:
        client.transition_model_version_stage(name=MODEL_NAME, version=version, stage="Production")
        print(f"🚀 {framework} promovido para Production (RMSE={rmse:.4f})")
    else:
        client.transition_model_version_stage(name=MODEL_NAME, version=version, stage="Staging")
        print(f"📌 {framework} mantido em Staging (RMSE={rmse})")
