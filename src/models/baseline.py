import argparse
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler


class MLP_PyTorch(nn.Module):
    def __init__(self, input_dim):
        super(MLP_PyTorch, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def treinar_pytorch(X_train, y_train, modelo_path="modelo_pytorch.pth"):
    input_dim = X_train.shape[1]
    model = MLP_PyTorch(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    for _ in range(50):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), modelo_path)
    return model


def carregar_pytorch(X_test, modelo_path="modelo_pytorch.pth"):
    input_dim = X_test.shape[1]
    model = MLP_PyTorch(input_dim)
    model.load_state_dict(torch.load(modelo_path, map_location=torch.device("cpu")))
    model.eval()

    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_pred = model(X_test_t).numpy()
    return y_pred


def carregar_dados_csv(csv_path: str) -> pd.DataFrame:
    print(f"Lendo dados do CSV: {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {csv_path}")

    try:
        dados = pd.read_csv(csv_path, header=[0, 1], index_col=0, parse_dates=True)
        if isinstance(dados.columns, pd.MultiIndex):
            dados.columns = dados.columns.get_level_values(0)
    except Exception:
        dados = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    if "Close" not in dados.columns:
        raise ValueError(f"A coluna 'Close' não foi encontrada no CSV. Colunas encontradas: {list(dados.columns)}")

    dados["Close"] = pd.to_numeric(dados["Close"], errors="coerce")
    dados = dados.dropna(subset=["Close"])

    return dados


def avaliar_modelo(y_real, y_pred, nome_modelo, scaler):
    y_real_inv = scaler.inverse_transform(y_real.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))

    mae = mean_absolute_error(y_real_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_real_inv, y_pred_inv))
    mape = np.mean(np.abs((y_real_inv - y_pred_inv) / y_real_inv)) * 100

    return {
        "Modelo": nome_modelo,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "MAPE (%)": round(mape, 2),
        "Predicoes": y_pred_inv,
    }



def main(args):
    ticker = args.ticker
    modelo_path = args.modelo
    start = args.start
    end = args.end
    janela = args.janela
    usar_keras = args.keras

    print(f"Avaliando {ticker} de {start} até {end} | janela={janela}")

    dados = carregar_dados_csv(args.data_path)
    precos = dados[["Close"]].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    precos_normalizados = scaler.fit_transform(precos)

    X, y = [], []
    for i in range(janela, len(precos_normalizados)):
        X.append(precos_normalizados[i - janela:i, 0])
        y.append(precos_normalizados[i, 0])

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        raise ValueError("Janela muito grande para o período escolhido.")

    X_keras = np.reshape(X, (X.shape[0], X.shape[1], 1))

    modelo_pytorch_path = args.modelo_pytorch
    if os.path.exists(modelo_pytorch_path):
        print("Carregando modelo PyTorch...")
        y_pred_torch = carregar_pytorch(X, modelo_pytorch_path)
    else:
        print("Treinando PyTorch...")
        treinar_pytorch(X, y, modelo_pytorch_path)
        y_pred_torch = carregar_pytorch(X, modelo_pytorch_path)

    modelo_sklearn_path = args.modelo_sklearn
    if os.path.exists(modelo_sklearn_path):
        print("Carregando Scikit...")
        mlp = joblib.load(modelo_sklearn_path)
    else:
        print("Treinando Scikit...")
        mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500)
        mlp.fit(X, y)
        joblib.dump(mlp, modelo_sklearn_path)

    y_pred_sklearn = mlp.predict(X).reshape(-1, 1)

    y_pred_keras = None
    if usar_keras:
        if modelo_path and os.path.exists(modelo_path):
            print(f"Carregando Keras... arquivo: {modelo_path}")
            modelo = tf.keras.models.load_model(modelo_path)
            y_pred_keras = modelo.predict(X_keras)
        else:
            print(f"Modelo Keras não encontrado: {modelo_path}")

    # ------------------ Baseline (naive) ------------------
    y_pred_baseline = X[:, -1].reshape(-1, 1)
    resultado_baseline = avaliar_modelo(y, y_pred_baseline, "Baseline", scaler)

    # ------------------ Avaliação ------------------
    resultados = []
    resultados.append(resultado_baseline)
    resultados.append(avaliar_modelo(y, y_pred_torch, "PyTorch", scaler))
    resultados.append(avaliar_modelo(y, y_pred_sklearn, "Scikit", scaler))

    if y_pred_keras is not None:
        resultados.append(avaliar_modelo(y, y_pred_keras, "Keras", scaler))

    y_pred_final = (y_pred_torch + y_pred_sklearn) / 2
    if y_pred_keras is not None:
        y_pred_final = (y_pred_torch + y_pred_sklearn + y_pred_keras) / 3

    resultados.append(avaliar_modelo(y, y_pred_final, "Ensemble", scaler))

    tabela = pd.DataFrame([{
        "Modelo": r["Modelo"],
        "MAE": r["MAE"],
        "RMSE": r["RMSE"],
        "MAPE (%)": r["MAPE (%)"],
    } for r in resultados])

    print("\n=== RESULTADOS ===")
    print(tabela)

    os.makedirs("reports", exist_ok=True)

    csv_path = "reports/metrics_comparison.csv"
    tabela.to_csv(csv_path, index=False)
    print(f"\nTabela salva em: {csv_path}")

    metrics_json = {}
    for r in resultados:
        nome = r["Modelo"].lower()
        nome = nome.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "")
        metrics_json[nome] = {
            "mae": float(r["MAE"]),
            "rmse": float(r["RMSE"]),
            "mape": float(r["MAPE (%)"]),
        }

    json_path = "reports/metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)

    print(f"Métricas salvas em: {json_path}")

    plt.figure(figsize=(12, 6))
    plt.plot(scaler.inverse_transform(y.reshape(-1, 1)), label="Real")

    for r in resultados:
        plt.plot(r["Predicoes"], label=r["Modelo"])

    plt.title(f"Comparação de modelos - {ticker}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino e avaliação de modelos")

    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--janela", type=int, default=90)

    parser.add_argument("--modelo", type=str, default=None, help="Modelo Keras (.keras)")
    parser.add_argument("--modelo-pytorch", type=str, default="modelo_pytorch.pth")
    parser.add_argument("--modelo-sklearn", type=str, default="modelo_sklearn.joblib")

    parser.add_argument("--keras", action="store_true", help="Ativar uso de modelo Keras")
    parser.add_argument("--data-path", type=str, default="data/raw/stock_data.csv", help="Caminho do CSV de entrada")

    args = parser.parse_args()
    main(args)
