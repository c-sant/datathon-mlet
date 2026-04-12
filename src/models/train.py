import argparse
import os
import yfinance as yf
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.tensorflow
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
import joblib
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Treino de modelos para previsão de ações")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker da ação (ex: ITUB4.SA)")
    parser.add_argument("--start", type=str, required=True, help="Data inicial no formato YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="Data final no formato YYYY-MM-DD")
    parser.add_argument("--janela", type=int, default=90, help="Janela de dias para previsão")
    parser.add_argument("--epochs", type=int, default=40, help="Número de épocas de treino")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamanho do batch para treino")
    parser.add_argument("--patience", type=int, default=5, help="Paciência para EarlyStopping no Keras")
    return parser.parse_args()

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

def treinar_pytorch(X_train, y_train, modelo_path):
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
    print(f"✅ Modelo PyTorch salvo em {modelo_path}")
    return model

def avaliar(y_true, y_pred, scaler):
    y_true_inv = scaler.inverse_transform(y_true.reshape(-1,1))
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1,1))
    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    mape = np.mean(np.abs((y_true_inv - y_pred_inv) / y_true_inv)) * 100
    return mae, rmse, mape

def preparar_dados(ticker, start, end, janela):
    df = yf.download(ticker, start=start, end=end)
    df = df[["Close"]].dropna()

    scaler = MinMaxScaler(feature_range=(0,1))
    dados = scaler.fit_transform(df.values)

    X, y = [], []
    for i in range(janela, len(dados)):
        X.append(dados[i-janela:i, 0])
        y.append(dados[i, 0])
    X, y = np.array(X), np.array(y)

    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return X_train, X_test, y_train, y_test, scaler

def main(args):
    ticker = args.ticker
    janela_dias = args.janela
    epocas = args.epochs
    batchsize = args.batch_size

    # prepara dados
    X_train, X_test, y_train, y_test, scaler = preparar_dados(ticker, args.start, args.end, janela_dias)

    mlflow.set_experiment("previsao_acoes")

    # PyTorch
    with mlflow.start_run(run_name="PyTorch"):
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("framework", "PyTorch")
        model_pytorch = treinar_pytorch(X_train, y_train, os.path.join(MODELS_DIR, f"modelo_{ticker}_pytorch.pth"))
        model_pytorch.eval()
        with torch.no_grad():
            y_pred_torch = model_pytorch(torch.tensor(X_test, dtype=torch.float32)).numpy()
        mae, rmse, mape = avaliar(y_test, y_pred_torch, scaler)
        mlflow.log_metric("MAE_pytorch", mae)
        mlflow.log_metric("RMSE_pytorch", rmse)
        mlflow.log_metric("MAPE_pytorch", mape)
        mlflow.pytorch.log_model(model_pytorch, "pytorch_model")

    # Sklearn
    with mlflow.start_run(run_name="Sklearn"):
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("framework", "Sklearn")
        mlp = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500)
        mlp.fit(X_train, y_train)
        joblib.dump(mlp, os.path.join(MODELS_DIR, f"modelo_{ticker}_sklearn.joblib"))
        y_pred_sklearn = mlp.predict(X_test).reshape(-1,1)
        mae, rmse, mape = avaliar(y_test, y_pred_sklearn, scaler)
        mlflow.log_metric("MAE_sklearn", mae)
        mlflow.log_metric("RMSE_sklearn", rmse)
        mlflow.log_metric("MAPE_sklearn", mape)
        mlflow.sklearn.log_model(mlp, "sklearn_model")

    # Keras
    with mlflow.start_run(run_name="Keras"):
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("framework", "Keras")
        modelo = Sequential([
            Input(shape=(X_train.shape[1], 1)),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])
        modelo.compile(optimizer='adam', loss='mean_squared_error')
        early_stop = EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True)
        modelo.fit(
            X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train,
            epochs=epocas,
            batch_size=batchsize,
            validation_data=(X_test.reshape(X_test.shape[0], X_test.shape[1], 1), y_test),
            callbacks=[early_stop]
        )
        modelo.save(os.path.join(MODELS_DIR, f"modelo_{ticker}.keras"))
        y_pred_keras = modelo.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1))
        mae, rmse, mape = avaliar(y_test, y_pred_keras, scaler)
        mlflow.log_metric("MAE_keras", mae)
        mlflow.log_metric("RMSE_keras", rmse)
        mlflow.log_metric("MAPE_keras", mape)
        mlflow.tensorflow.log_model(modelo, "keras_model")

if __name__ == "__main__":
    args = parse_args()
    main(args)
