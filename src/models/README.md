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


ingest → train → baseline


---

## 📥 ingest

Responsável por:

- Coletar dados via `yfinance`
- Persistir dataset em:


data/raw/stock_data.csv


📌 Script:


data/ingest.py


---

## 🧠 train

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


---

# 📂 Estrutura do Projeto

```bash
datathon-mlet/
│
├── data/
│   ├── ingest.py                # Script de ingestão (yfinance → CSV)
│   └── raw/
│       └── stock_data.csv       # Dataset gerado (controlado pelo DVC)
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
│
├── src/
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
├── dvc.yaml                   # Pipeline (ingest → train → baseline → plots)
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

Lê dados do CSV gerado no ingest
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
  --keras \
  --data-path data/raw/stock_data.csv

🧪 MLflow (nível profissional - SQLite)
🔹 Estrutura
mlflow/
├── mlflow.db
└── artifacts/

🔹 Subir interface

mlflow ui \
  --backend-store-uri sqlite:///mlflow/mlflow.db \
  --default-artifact-root ./mlflow/artifacts \
  --port 5000

Acesse:

http://127.0.0.1:5000

📊 Métricas utilizadas
MAE — Erro Absoluto Médio
RMSE — Raiz do Erro Quadrático Médio
MAPE — Erro Percentual Médio

👉 menor = melhor

# 💼 Métricas de negócio mapeadas para métricas técnicas

Para avaliar o impacto real dos modelos no contexto de negócio, as métricas técnicas foram traduzidas em indicadores financeiros e de risco.

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

Exemplo real do projeto:

- Baseline: **MAE = 0.456**
- Keras: **MAE = 1.112**

📌 Interpretação:
- O baseline erra em média **R$ 0,456 por ação**
- O Keras erra em média **R$ 1,112 por ação**

👉 Em uma operação com 1.000 ações:

- Baseline → erro médio ≈ **R$ 456**
- Keras → erro médio ≈ **R$ 1.112**

⚠️ Diferença de **R$ 656 por operação**

---

### 🔹 RMSE → Risco de erro grande

O RMSE penaliza erros maiores, sendo um indicador de risco.

Exemplo:

- Baseline: **RMSE = 0.621**
- PyTorch: **RMSE = 2.634**

📌 Interpretação:
- PyTorch tem maior probabilidade de gerar erros extremos
- Baseline é mais estável

👉 Em cenários de alta volatilidade, isso reduz risco financeiro

---

### 🔹 MAPE → Precisão percentual

O MAPE mostra o erro relativo ao valor da ação.

Exemplo:

- Baseline: **MAPE = 1.14%**
- Scikit: **MAPE = 4.87%**

📌 Interpretação:
- Baseline erra ~1% do valor da ação
- Scikit erra quase 5%

👉 Quanto menor o MAPE, mais confiável é o modelo em diferentes faixas de preço

---

# 📊 Visualização dos resultados

Os gráficos gerados automaticamente ajudam a interpretar rapidamente o desempenho dos modelos:

### 📊 Comparação geral
![Comparação](reports/comparacao_modelos.png)

- Mostra todas as métricas juntas
- Permite identificar rapidamente o melhor modelo

---

### 🏆 Ranking por MAE
![Ranking](reports/ranking_mae.png)

- Ordena modelos pelo erro médio
- Facilita decisão de negócio

---

### 📉 Erro percentual (MAPE)
![MAPE](reports/mape.png)

- Mostra impacto proporcional
- Importante para comparação entre ativos

---

# 📈 Exemplos de ganho e perda

## ✅ Cenário ideal (baseline)

- MAE baixo → menor erro financeiro
- RMSE baixo → menor risco
- MAPE baixo → maior precisão

👉 Decisões mais confiáveis

---

## ⚠️ Cenário de risco (modelos complexos)

Exemplo: PyTorch

- MAE alto → maior erro médio
- RMSE alto → risco de grandes perdas
- MAPE alto → baixa precisão

👉 Pode gerar decisões incorretas em operações reais

---

## 💡 Comparação prática

| Modelo   | Erro médio (R$) | Risco | Precisão |
|----------|----------------|-------|-----------|
| Baseline | Baixo          | Baixo | Alta      |
| Keras    | Médio          | Médio | Boa       |
| Scikit   | Alto           | Alto  | Média     |
| PyTorch  | Alto           | Alto  | Baixa     |

---

# 🎯 Conclusão de negócio

Apesar do uso de modelos avançados, o **baseline naive apresentou o melhor desempenho**.

Isso indica que a série possui forte dependência temporal, onde o comportamento recente é altamente preditivo.

👉 Em termos de negócio:

- menor erro financeiro
- menor risco de perdas
- maior confiabilidade nas previsões

---

# 🧠 Insight chave

> Modelos mais complexos nem sempre geram mais valor.  
> Em séries financeiras, simplicidade pode ser a melhor estratégia.

📊 Resultados principais

| Modelo       | MAE       | RMSE      | MAPE (%) |
|--------------|-----------|-----------|----------|
| **Baseline** | **0.456** | **0.621** | **1.14** |
| Keras        | 1.112     | 1.567     | 2.73     |
| Ensemble     | 1.648     | 2.086     | 4.09     |
| Scikit       | 1.973     | 2.581     | 4.87     |
| PyTorch      | 2.185     | 2.634     | 5.49     |

🧠 Insights
O baseline (naive) foi o melhor modelo
A série apresenta forte autocorrelação temporal
Modelos complexos não superaram a abordagem simples
Keras foi o melhor entre os modelos de ML/DL
Ensemble não trouxe ganho relevante
PyTorch apresentou pior desempenho

⚠️ Evolução importante: eliminação de SPOF

Antes:

Notebook compartilhado (SPOF)

Agora:

Scripts versionados (ingest, train, baseline)
Pipeline automatizado com DVC

✔ Reprodutibilidade
✔ Execução determinística
✔ Menos erro humano

📦 Setup do Ambiente

🔹 Usando pyenv (recomendado)

pyenv install 3.13.0
pyenv virtualenv 3.13.0 datathon-env
pyenv activate datathon-env

🔹 Ou usando venv

python -m venv venv
source venv/bin/activate
🔹 Instalar dependências
pip install -e .
pip install -e ".[dev]"

📦 Pipeline versionado (DVC + Docker)

✔ DVC
dvc repro

✔ Métricas
dvc metrics show
dvc metrics diff

🐳 Docker

# Execução via Docker
Docker Desktop instalado
Docker em execução
(WSL integrado, caso esteja usando Linux via Windows)

🔹 Build

Geral:
docker build -t datathon-mlet .

Pipeline:
docker compose build pipeline

MLFLOW:
docker compose build mlflow

🔹 Executar pipeline

docker run --rm -it \
  -u $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e USER=user \
  -v $(pwd):/app \
  datathon-mlet
🔹 Console interativo

docker run --rm -it \
  -u $(id -u):$(id -g) \
  -e HOME=/tmp \
  -e USER=user \
  -v $(pwd):/app \
  datathon-mlet bash

📦 Gestão de dependências

Uso de pyproject.toml:

✔ padrão moderno Python
✔ substitui requirements.txt
✔ integração com Docker e DVC


🚀 Status

✔ Pipeline funcional
✔ Métricas registradas
✔ MLflow integrado
✔ DVC configurado
✔ Docker reproduzível

🧠 Aprendizado chave

Modelos simples podem superar modelos complexos em séries temporais.

🚀 Diferenciais do projeto

✔ Pipeline MLOps completo
✔ Reprodutibilidade (DVC + Docker)
✔ Comparação com baseline real
✔ Multi-framework (PyTorch + Keras)
✔ Métricas versionadas