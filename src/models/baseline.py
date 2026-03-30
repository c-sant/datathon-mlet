import argparse
import os
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
import joblib

# ------------------ PyTorch ------------------
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
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1,1)

    for epoch in range(50):
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
    model.load_state_dict(torch.load(modelo_path))
    model.eval()

    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_pred = model(X_test_t).numpy()
    return y_pred

# ------------------ Avaliação de métricas ------------------
def avaliar_modelo(y_real, y_pred, nome_modelo, scaler):
    y_real_inv = scaler.inverse_transform(y_real.reshape(-1,1))
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1,1))

    mae = mean_absolute_error(y_real_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_real_inv, y_pred_inv))
    mape = np.mean(np.abs((y_real_inv - y_pred_inv) / y_real_inv)) * 100

    return {
        "Modelo": nome_modelo,
        "MAE": round(mae,3),
        "RMSE": round(rmse,3),
        "MAPE (%)": round(mape,2),
        "Predicoes": y_pred_inv
    }

# ------------------ Principal ------------------
def main(args):
    ticker = args.ticker
    modelo_path = args.modelo
    print(f"Avaliando {ticker}...")

    # Coletar dados
    dados = yf.download(ticker, start="2025-01-01", end="2025-12-31")
    precos = dados[['Close']].values

    # Normalizar
    scaler = MinMaxScaler(feature_range=(0,1))
    precos_normalizados = scaler.fit_transform(precos)

    janela = 90
    X, y = [], []
    for i in range(janela, len(precos_normalizados)):
        X.append(precos_normalizados[i-janela:i, 0])
        y.append(precos_normalizados[i, 0])
    X = np.array(X)
    y = np.array(y)

    # Reshape para Keras
    X_keras = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # ------------------ PyTorch ------------------
    modelo_pytorch_path = "modelo_pytorch.pth"
    if os.path.exists(modelo_pytorch_path):
        print("Carregando modelo PyTorch salvo...")
        y_pred_torch = carregar_pytorch(X, modelo_pytorch_path)
    else:
        print("Treinando novo modelo PyTorch...")
        model = treinar_pytorch(X, y, modelo_pytorch_path)
        y_pred_torch = carregar_pytorch(X, modelo_pytorch_path)

    # ------------------ Scikit-Learn ------------------
    modelo_sklearn_path = "modelo_sklearn.joblib"
    if os.path.exists(modelo_sklearn_path):
        print("Carregando modelo Scikit-Learn salvo...")
        mlp = joblib.load(modelo_sklearn_path)
    else:
        print("Treinando novo modelo Scikit-Learn...")
        mlp = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500)
        mlp.fit(X, y)
        joblib.dump(mlp, modelo_sklearn_path)
    y_pred_sklearn = mlp.predict(X).reshape(-1,1)

    # ------------------ Keras (fallback) ------------------
    if os.path.exists(modelo_path):
        print("Carregando modelo Keras salvo...")
        modelo = tf.keras.models.load_model(modelo_path)
        y_pred_keras = modelo.predict(X_keras)
    else:
        y_pred_keras = None

    # ------------------ Avaliar cada modelo ------------------
    resultados = []
    resultados.append(avaliar_modelo(y, y_pred_torch, "PyTorch", scaler))
    resultados.append(avaliar_modelo(y, y_pred_sklearn, "Scikit-Learn", scaler))
    if y_pred_keras is not None:
        resultados.append(avaliar_modelo(y, y_pred_keras, "Keras", scaler))

    # ------------------ Ensemble final ------------------
    y_pred_final = (y_pred_torch + y_pred_sklearn) / 2
    if y_pred_keras is not None:
        y_pred_final = (y_pred_final + y_pred_keras) / 2
    resultados.append(avaliar_modelo(y, y_pred_final, "Ensemble Final", scaler))

    # ------------------ Tabela comparativa ------------------
    tabela_metricas = pd.DataFrame([{
        "Modelo": r["Modelo"],
        "MAE": r["MAE"],
        "RMSE": r["RMSE"],
        "MAPE (%)": r["MAPE (%)"]
    } for r in resultados])

    fig, ax = plt.subplots(figsize=(8,4))
    ax.axis('off')
    ax.table(cellText=tabela_metricas.values,
             colLabels=tabela_metricas.columns,
             loc='center')
    plt.title("Tabela comparativa de métricas")
    plt.show()

    # ------------------ Gráfico comparativo ------------------
    plt.figure(figsize=(12,6))
    plt.plot(scaler.inverse_transform(y.reshape(-1,1)), label="Preço Real")
    for r in resultados:
        plt.plot(r["Predicoes"], label=r["Modelo"])
    plt.title(f"Comparação de previsões - {ticker}")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline de previsão de ações")
    parser.add_argument("--modelo", type=str, required=True, help="Caminho do modelo salvo (.keras)")
    parser.add_argument("--ticker", type=str, default="ITUB4.SA", help="Ticker para coletar dados atuais")
    args = parser.parse_args()
    main(args)