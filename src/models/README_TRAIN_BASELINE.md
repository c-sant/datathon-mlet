# 📊 Projeto Fase 5 – Pipeline de Previsão de Ações com MLOps

## 🧠 Objetivo

Desenvolver um pipeline de previsão de preços de ações utilizando:

- Machine Learning
- Deep Learning
- Comparação com baseline
- Tracking de experimentos com MLflow
- Pipeline reprodutível com DVC
- Execução padronizada com Docker

---

# ⚙️ Arquitetura do Projeto

## 🔁 Pipeline


ingest → feature_engineering → train → baseline → plots


---

## 📥 ingest

Responsável por:

- Coletar dados via `yfinance`
- Persistir dataset em:


data/raw/stock_data.csv


📌 Script:


data/ingest.py

---

## 🧩 feature_engineering

Responsável por:

- Criação de features estatísticas (EDA)
- Tratamento de outliers
- Remoção de correlação
- Redução de multicolinearidade (VIF)
- Seleção automática de features
- Escalonamento (MinMaxScaler)
- Redução de dimensionalidade (PCA)

Gera:

data/raw/stock_features.csv  
models/scaler_features.joblib  
models/pca_features.joblib  

Relatórios:

reports/correlation_matrix.csv  
reports/vif_report.csv  
reports/selected_features.csv  
reports/pca_explained_variance.csv  

📌 Script:

src/features/feature_engineering.py

---

## 🧠 train

👉 Utiliza automaticamente o dataset tratado: data/raw/stock_features.csv

Treina múltiplos modelos:

- Baseline (Naive)
- Scikit-Learn (MLP)
- PyTorch (MLP)
- Keras (LSTM)

Salva modelos em:


models/


Registra experimentos no MLflow:


mlflow/mlflow.db


---

## 📊 baseline

Avalia todos os modelos:

- Baseline
- PyTorch
- Scikit
- Keras
- Ensemble

Gera:

```bash
reports/metrics.json
reports/metrics_comparison.csv
reports/model_predictions_comparison.png


---

# 📂 Estrutura do Projeto

```bash
datathon-mlet/
│
├── data/
│   ├── ingest.py                # Script de ingestão (yfinance → CSV)
│   └── raw/
│       ├── stock_data.csv       # Dataset gerado (controlado pelo DVC)
│       └── stock_features.csv   # Dataset com features tratadas
│
├── models/                      # Modelos treinados
│   ├── modelo_ITUB4.SA_pytorch.pth
│   ├── modelo_ITUB4.SA_sklearn.joblib
│   └── modelo_ITUB4.SA.keras
│
├── mlflow/
│   ├── mlflow.db               # Banco SQLite de experimentos
│   └── artifacts/              # Artefatos dos modelos
│
├── reports/
│   ├── metrics.json            # Métricas versionadas (DVC)
│   ├── metrics_comparison.csv  # Tabela comparativa
│   ├── comparacao_modelos.png  # 📊 Gráfico 1
│   ├── ranking_mae.png         # 📊 Gráfico 2
│   └── mape.png                # 📊 Gráfico 3
│   ├── model_predictions_comparison.png
│   ├── correlation_matrix.csv
│   ├── vif_report.csv
│   ├── selected_features.csv
│   └── pca_explained_variance.csv
│
├── src/
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── train.py            # Pipeline de treino + MLflow
│   │   └── baseline.py         # Avaliação + geração de métricas
│   │
│   └── utils/
│       └── plot_metrics.py     # Geração automática de gráficos
│
├── .dvc/                       # Cache do DVC
├── .dvcignore
│
├── dvc.yaml                   # Pipeline (ingest → feature_engineering → train → baseline → plots)
├── params.yaml                # Parâmetros do pipeline
│
├── pyproject.toml             # Dependências (padrão moderno)
├── Dockerfile                 # Container reprodutível
│
└── README.md                  # Documentação do projeto

⚙️ Parâmetros disponíveis
Parâmetro	Descrição
--ticker	Código do ativo (ex: ITUB4.SA)
--start	Data inicial
--end	Data final
--janela	Janela temporal
--epochs	Épocas (DL)
--batch	Batch size
--keras	Ativa modelo Keras

🧠 Baseline

O baseline.py executa:

Baseline (naive)
PyTorch
Scikit-Learn
Keras
Ensemble

▶️ Executar baseline

python src/models/baseline.py \
  --ticker ITUB4.SA \
  --start 2025-04-01 \
  --end 2026-04-30 \
  --janela 90 \
  --modelo models/modelo_ITUB4.SA.keras \
  --modelo-pytorch models/modelo_ITUB4.SA_pytorch.pth \
  --modelo-sklearn models/modelo_ITUB4.SA_sklearn.joblib \
  --keras

🧠 Treinamento (com MLflow)

O train.py:

Utiliza automaticamente o dataset tratado gerado pelo feature_engineering
Lê dados de data/raw/stock_features.csv
Cria janelas temporais
Treina modelos:
Baseline Naive
Scikit-Learn
PyTorch
(Opcional) Keras
Registra tudo no MLflow (SQLite)

▶️ Executar treino

python src/models/train.py \
  --ticker ITUB4.SA \
  --start 2025-04-01 \
  --end 2026-04-30 \
  --janela 90 \
  --epochs 40 \
  --batch 32 \
  --keras 

🧪 MLflow (nível profissional - SQLite)
🔹 Estrutura
mlflow/
├── mlflow.db
└── artifacts/

🔹 Subir interface

mlflow ui \
  --backend-store-uri sqlite:///mlflow/mlflow.db \
  --default-artifact-root ./mlflow/artifacts \
  --host 0.0.0.0 \
  --port 5000

Acesse:

http://127.0.0.1:5000

📊 Métricas utilizadas
MAE — Erro Absoluto Médio
RMSE — Raiz do Erro Quadrático Médio
MAPE — Erro Percentual Médio

👉 menor = melhor

# 💼 Métricas de negócio mapeadas para métricas técnicas

Para avaliar o impacto real dos modelos no contexto de negócio, as métricas técnicas foram traduzidas em indicadores financeiros e de risco com base nos resultados obtidos no pipeline.

---

## 📊 Métricas técnicas utilizadas

- **MAE (Mean Absolute Error)** → erro médio absoluto  
- **RMSE (Root Mean Squared Error)** → penaliza erros maiores  
- **MAPE (Mean Absolute Percentage Error)** → erro percentual médio  

👉 Quanto menor, melhor o desempenho do modelo.

---

## 🔗 Tradução para métricas de negócio

### 🔹 MAE → Impacto financeiro direto

O MAE representa o erro médio em reais por ação.

Resultados reais do projeto:

- PyTorch: **MAE = 0.504**
- Baseline: **MAE = 0.763**
- Keras: **MAE = 0.921**
- Ensemble: **MAE = 1.239**
- Scikit: **MAE = 2.806**

📌 Interpretação:

👉 Em uma operação com 1.000 ações:

- PyTorch → erro médio ≈ **R$ 504**
- Baseline → erro médio ≈ **R$ 763**
- Scikit → erro médio ≈ **R$ 2.806**

⚠️ Diferença de até **R$ 2.300 por operação** entre modelos

---

### 🔹 RMSE → Risco de erro grande

O RMSE indica a presença de erros extremos (outliers).

Resultados:

- PyTorch: **RMSE = 1.088**
- Baseline: **RMSE = 1.009**
- Keras: **RMSE = 1.205**
- Ensemble: **RMSE = 2.824**
- Scikit: **RMSE = 7.45**

📌 Interpretação:

- Scikit apresenta altíssimo risco de erro extremo
- Ensemble também apresenta instabilidade
- Baseline e PyTorch são mais estáveis

👉 Em cenários reais, isso impacta diretamente o risco financeiro

---

### 🔹 MAPE → Precisão percentual

O MAPE mostra o erro proporcional ao valor da ação.

Resultados:

- PyTorch: **MAPE = 1.14%**
- Baseline: **MAPE = 1.8%**
- Keras: **MAPE = 2.2%**
- Ensemble: **MAPE = 2.79%**
- Scikit: **MAPE = 6.26%**

📌 Interpretação:

- PyTorch é o modelo mais preciso proporcionalmente
- Scikit apresenta baixa confiabilidade
- Baseline mantém boa robustez

---

# 📊 Visualização dos resultados

Os gráficos são gerados automaticamente pelo estágio `plots` do pipeline DVC.

---

### 📊 Comparação geral
![Comparação](reports/comparacao_modelos.png)

---

### 🏆 Ranking por MAE
![Ranking](reports/ranking_mae.png).

---

### 📉 Erro percentual (MAPE)
![MAPE](reports/mape.png)

---

### 🔍 Comparação de predições (Real vs Modelos)
![Predições](reports/model_predictions_comparison.png)

📌 Observação importante:

- O modelo **Scikit** apresenta grande divergência no final da série (instabilidade)
- O **PyTorch** acompanha melhor a tendência geral
- O **Baseline** mantém comportamento consistente

---

# 📈 Exemplos de ganho e perda

## ✅ Cenário ideal (PyTorch / Baseline)

- MAE baixo → menor erro financeiro
- RMSE baixo → menor risco
- MAPE baixo → maior precisão

👉 Decisões mais confiáveis

---

## ⚠️ Cenário de risco (Scikit)

- MAE muito alto → alto erro financeiro
- RMSE extremamente alto → risco de perdas severas
- MAPE elevado → baixa precisão

👉 Pode gerar prejuízo significativo em operações reais

---

## 💡 Comparação prática

| Modelo   | Erro médio (R$) | Risco      | Precisão  |
|----------|-----------------|------------|-----------|
| PyTorch  | Muito baixo     | Baixo      | Alta      |
| Baseline | Baixo           | Baixo      | Boa       |
| Keras    | Médio           | Médio      | Boa       |
| Ensemble | Alto            | Alto       | Média     |
| Scikit   | Muito alto      | Muito alto | Baixa     |

---

# 🎯 Conclusão de negócio

Diferente da hipótese inicial, o melhor modelo foi:

👉 **PyTorch**

Isso indica:

- O uso de múltiplas features (feature engineering) agregou valor
- O modelo conseguiu capturar melhor a dinâmica da série

👉 Em termos de negócio:

- menor erro financeiro
- maior precisão percentual
- melhor aderência à série real

---

# 🧠 Insight chave

> Feature engineering bem aplicado + modelo adequado supera baseline simples.

📊 Resultados principais

| Modelo       | MAE       | RMSE      | MAPE (%) |
|--------------|-----------|-----------|----------|
| **PyTorch**  | **0.504** | **1.088** | **1.14** |
| Baseline     | 0.763     | 1.009     | 1.8      |
| Keras        | 0.921     | 1.205     | 2.2      |
| Ensemble     | 1.239     | 2.824     | 2.79     |
| Scikit       | 2.806     | 7.45      | 6.26     |

---

## 🧠 Insights

- PyTorch apresentou melhor desempenho geral
- Feature engineering foi decisivo
- Scikit demonstrou instabilidade severa
- Ensemble não trouxe ganho significativo
- Baseline ainda é forte benchmark

⚠️ Evolução importante: eliminação de SPOF

Antes:

Notebook compartilhado (SPOF)

Agora:

Scripts versionados (ingest, train, baseline)
Pipeline automatizado com DVC

✔ Reprodutibilidade
✔ Execução determinística
✔ Menos erro humano

# 📦 Setup do Ambiente

## 🔹 Usando pyenv (recomendado)

```bash
pyenv install 3.13.0
pyenv virtualenv 3.13.0 datathon-env
pyenv local datathon-env
pyenv activate datathon-env
🔹 Ou usando venv

Linux/macOS:

python -m venv venv
source venv/bin/activate

Windows:

venv\Scripts\activate
🔹 Instalar dependências
pip install -e .
pip install -e ".[dev]"
pip install -e ".[test]"

📦 Pipeline versionado (DVC + Docker)
✔ DVC
dvc repro

Forçar execução completa:

dvc repro -f
✔ Métricas
dvc metrics show
dvc metrics diff
🐳 Docker

Pré-requisitos:

Docker Desktop instalado
Docker em execução
WSL integrado (caso esteja usando Windows + Linux)

🔹 Build
docker compose build

🔹 Executar pipeline
docker compose run --rm pipeline

🔹 Executar testes
docker compose run --rm test

🔹 Subir MLflow
docker compose up mlflow

📦 Gestão de dependências

Uso de pyproject.toml:

✔ padrão moderno Python
✔ substitui requirements.txt
✔ integração com Docker e DVC

🚀 Status

✔ Pipeline funcional
✔ Feature engineering avançado
✔ Pipeline automático com DVC
✔ MLflow integrado
✔ Docker reproduzível
✔ Testes automatizados

🧠 Aprendizado chave

Feature engineering + escolha correta de modelo pode superar abordagens simples.

🚀 Diferenciais do projeto

✔ Pipeline MLOps completo
✔ Reprodutibilidade (DVC + Docker)
✔ Feature engineering avançado
✔ Comparação com baseline real
✔ Multi-framework (PyTorch + Keras)
✔ Métricas versionadas
✔ Pipeline automatizado end-to-end
