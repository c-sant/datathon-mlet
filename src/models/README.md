# 📊 Previsão de Preços de Ações


![Python](https://img.shields.io/badge/Python-3.13-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange))

---

## ⚠️ Compatibilidade

Este projeto foi desenvolvido utilizando **Python 3.13**.

Algumas bibliotecas (ex: TensorFlow) podem apresentar limitações em versões recentes do Python.  
Caso encontre problemas, recomenda-se utilizar Python 3.11 como fallback.

--- 

## 📦 Setup do ambiente

```bash
# criar ambiente com Python 3.13
pyenv install 3.13.0
pyenv virtualenv 3.13.0 datathon-env
pyenv local datathon-env

# instalar dependências
pip install --upgrade pip
pip install -r requirements.txt

---

## 🧪 (Opcional mas forte) — verificação

```markdown
## 🔍 Verificação do ambiente

```bash
python --version
# esperado: Python 3.13.x

---

## 🧠 Sobre o projeto

Este projeto realiza previsão de preços de ativos financeiros utilizando modelos de Machine Learning e Deep Learning, com execução via linha de comando e rastreamento de experimentos com MLflow.


---

## 🎯 Objetivo

- Construir pipeline de previsão de séries temporais
- Comparar modelos (Baseline, Scikit, PyTorch, Keras)
- Registrar métricas automaticamente no MLflow
- Permitir execução parametrizada via CLI

---

## 📁 Estrutura do projeto

```bash
src/
└── models/
    ├── baseline.py   # Avaliação e comparação de modelos
    └── train.py      # Pipeline de treino + MLflow (SQLite)

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

PyTorch
Scikit-Learn
Ensemble

▶️ Executar baseline
python src/models/baseline.py \
--ticker ITUB4.SA \
--start 2025-04-01 \
--end 2026-04-30 \
--janela 90

🧠 Treinamento (com MLflow)

O train.py:

Baixa dados via Yahoo Finance
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
--batch 32

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

📊 Métricas avaliadas
MAE — Erro Absoluto Médio
RMSE — Raiz do Erro Quadrático Médio
MAPE — Erro Percentual Médio

📊 Resultados dos Modelos
Modelo	MAE ↓	RMSE ↓	MAPE (%) ↓
PyTorch	1.781	2.457	4.42
Scikit-Learn	1.038	1.353	2.67
Ensemble	1.316	1.777	3.31

📌 Menores valores indicam melhor desempenho
🏆 Melhor modelo: Scikit-Learn

🧠 Insights
Modelos simples podem superar redes neurais em séries financeiras
Ensemble nem sempre melhora desempenho
Janela temporal impacta diretamente o resultado
MLflow permite rastreabilidade completa dos experimentos

✅ Requisitos atendidos
✔ Baseline treinado
✔ Métricas registradas no MLflow
✔ Pipeline parametrizado
✔ Comparação entre modelos
✔ Tracking persistente (SQLite)

📦 Setup do ambiente
pyenv virtualenv 3.13 datathon-env
pyenv local datathon-env

pip install -r requirements.txt
👨‍💻 Autor

Claudio
Data Engineer / DBA