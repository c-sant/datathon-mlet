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
import tensorflow as tf
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

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

def main(args):
    ticker = args.ticker
    start_date = args.start
    end_date = args.end
    janela_dias = args.janela
    epocas = args.epochs
    batchsize = args.batch

    print(f"Treinando modelos para {ticker} de {start_date} até {end_date}")

    # Definir pasta raiz para salvar modelos
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    MODELS_DIR = os.path.join(ROOT_DIR, "models")
    print(f"📂 Modelos serão salvos em: {MODELS_DIR}")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Coleta de dados
    dados = yf.download(ticker, start=start_date, end=end_date)
    precos = dados[['Close']].values

    # 2. Normalização
    scaler = MinMaxScaler(feature_range=(0,1))
    precos_normalizados = scaler.fit_transform(precos)

    # 3. Criar sequências
    X, y = [], []
    for i in range(janela_dias, len(precos_normalizados)):
        X.append(precos_normalizados[i-janela_dias:i, 0])
        y.append(precos_normalizados[i, 0])
    X = np.array(X)
    y = np.array(y)

    # Reshape para Keras
    X_keras = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split treino/teste
    tamanho_treino = int(len(X) * 0.8)
    X_train, X_test = X[:tamanho_treino], X[tamanho_treino:]
    y_train, y_test = y[:tamanho_treino], y[tamanho_treino:]

    # ------------------ MLflow ------------------
    mlflow.set_experiment("previsao_acoes")

    with mlflow.start_run():
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("janela", janela_dias)
        mlflow.log_param("epochs", epocas)
        mlflow.log_param("batch_size", batchsize)

        # PyTorch
        modelo_pytorch_path = os.path.join(MODELS_DIR, f"modelo_{ticker}_pytorch.pth")
        model_pytorch = treinar_pytorch(X_train, y_train, modelo_pytorch_path)
        model_pytorch.eval()
        with torch.no_grad():
            y_pred_torch = model_pytorch(torch.tensor(X_test, dtype=torch.float32)).numpy()
        mae_torch, rmse_torch, mape_torch = avaliar(y_test, y_pred_torch, scaler)
        print(f"[PyTorch] MAE={mae_torch:.2f}, RMSE={rmse_torch:.2f}, MAPE={mape_torch:.2f}%")
        mlflow.log_metric("MAE_pytorch", mae_torch)
        mlflow.log_metric("RMSE_pytorch", rmse_torch)
        mlflow.log_metric("MAPE_pytorch", mape_torch)

        # Regitra no Model Registry
        mlflow.pytorch.log_model(model_pytorch, "pytorch_model")

        # Scikit-Learn
        modelo_sklearn_path = os.path.join(MODELS_DIR, f"modelo_{ticker}_sklearn.joblib")
        mlp = MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500)
        mlp.fit(X_train, y_train)
        joblib.dump(mlp, modelo_sklearn_path)
        print(f"✅ Modelo Scikit-Learn salvo em {modelo_sklearn_path}")
        
        # Regitra no Model Registry
        mlflow.sklearn.log_model(mlp, "sklearn_model")

        y_pred_sklearn = mlp.predict(X_test).reshape(-1,1)
        mae_sklearn, rmse_sklearn, mape_sklearn = avaliar(y_test, y_pred_sklearn, scaler)
        print(f"[Scikit-Learn] MAE={mae_sklearn:.2f}, RMSE={rmse_sklearn:.2f}, MAPE={mape_sklearn:.2f}%")
        mlflow.log_metric("MAE_sklearn", mae_sklearn)
        mlflow.log_metric("RMSE_sklearn", rmse_sklearn)
        mlflow.log_metric("MAPE_sklearn", mape_sklearn)
        mlflow.sklearn.log_model(mlp, "sklearn_model")

        # Keras opcional
        if args.keras:
            print("Treinando Keras (opcional)...")
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
            modelo.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train,
                       epochs=epocas,
                       batch_size=batchsize,
                       validation_data=(X_test.reshape(X_test.shape[0], X_test.shape[1], 1), y_test),
                       callbacks=[early_stop])
            modelo_keras_path = os.path.join(MODELS_DIR, f"modelo_{ticker}.keras")
            modelo.save(modelo_keras_path)
            print(f"✅ Modelo Keras salvo em {modelo_keras_path}")

            # Regitra no Model Registry
            mlflow.tensorflow.log_model(modelo, "keras_model")

            y_pred_keras = modelo.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1))
            mae_keras, rmse_keras, mape_keras = avaliar(y_test, y_pred_keras, scaler)
            print(f"[Keras] MAE={mae_keras:.2f}, RMSE={rmse_keras:.2f}, MAPE={mape_keras:.2f}%")
            mlflow.log_metric("MAE_keras", mae_keras)
            mlflow.log_metric("RMSE_keras", rmse_keras)
            mlflow.log_metric("MAPE_keras", mape_keras)
            mlflow.tensorflow.log_model(modelo, "keras_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinar modelos para previsão de preços")
    parser.add_argument("--ticker", type=str, default="ITUB4.SA", help="Código do ativo")
    parser.add_argument("--start", type=str, default="2024-01-01", help="Data inicial")
    parser.add_argument("--end", type=str, default="2025-12-31", help="Data final")
    parser.add_argument("--janela", type=int, default=90, help="Tamanho da janela de dias")
    parser.add_argument("--epochs", type=int, default=40, help="Número de épocas")
    parser.add_argument("--batch", type=int, default=32, help="Tamanho do batch")
    parser.add_argument("--patience", type=int, default=4, help="Número de épocas sem melhora antes de parar o treino")
    parser.add_argument("--keras", action="store_true", help="Treinar também modelo Keras como fallback")
    args = parser.parse_args()
    main(args)
    