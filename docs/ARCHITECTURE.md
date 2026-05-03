# 🏗️ Arquitetura do Sistema - Datathon MLet

**Versão:** 1.0  
**Data:** Maio 2026  
**Projeto:** Previsão de Ações com MLOps, RAG e Inteligência Artificial

---

## 📋 Sumário Executivo

O **Datathon MLet** é uma plataforma integrada de aprendizado de máquina e inteligência artificial para previsão de ações em bolsa de valores. O sistema combina técnicas modernas de MLOps, sistemas de Geração Aumentada por Recuperação (RAG) e múltiplos frameworks de ML em um pipeline reprodutível, containerizado e totalmente operacionalizável.

### 🎯 Principais Características

- ✅ **Pipeline Reprodutível**: DVC garante rastreabilidade de dados e modelos
- ✅ **Múltiplos Frameworks**: PyTorch, TensorFlow/Keras, Scikit-Learn
- ✅ **MLOps Integrado**: MLflow para tracking de experimentos e versionamento
- ✅ **Sistema RAG**: Geração de respostas contextualizado com recuperação de informação
- ✅ **API Moderna**: FastAPI com uvicorn para serving de modelos
- ✅ **Containerização Completa**: Docker + Docker Compose para ambiente consistente
- ✅ **Escalabilidade**: Preparado para deployment em nuvem

---

## 🏛️ Arquitetura de Componentes

### Visão em Camadas

```
┌─────────────────────────────────────────────────────────────┐
│                    APRESENTAÇÃO/API                         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ FastAPI + Uvicorn (Porta 8000)                         │ │
│  │ - Endpoints para previsão de ações                     │ │
│  │ - Integração RAG para análise contextual               │ │
│  │ - Health checks e métricas                             │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│              ORQUESTRAÇÃO & MONITORING                       │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ MLflow Tracking (Porta 5000)                           │ │
│  │ - Dashboard de experimentos                            │ │
│  │ - Versionamento de modelos                             │ │
│  │ - Comparação de métricas (MAE, RMSE, MAPE)             │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ DVC Pipeline Orchestration                             │ │
│  │ - Versionamento de datasets                            │ │
│  │ - Rastreabilidade de estágios                          │ │
│  │ - Reprodutibilidade garantida                          │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│              CAMADA DE APLICAÇÃO                             │
│  ┌──────────────────┐  ┌──────────────────┐                 │
│  │ Sistema RAG      │  │ Feature Pipeline │                 │
│  ├──────────────────┤  ├──────────────────┤                 │
│  │ - SentenceTransf │  │ - Normalização   │                 │
│  │ - FAISS Vector   │  │ - PCA            │                 │
│  │ - LLM Generator  │  │ - Feature Select │                 │
│  │ - Answering QA   │  │ - Scaling        │                 │
│  └──────────────────┘  └──────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│              CAMADA DE MODELOS DE ML                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ PyTorch LSTM │  │ TensorFlow   │  │ Scikit-Learn │       │
│  │ - Séries     │  │ - RNN/Dense  │  │ - XGBoost    │       │
│  │   Temporais  │  │ - Conv Nets  │  │ - Random     │       │
│  │ - GPUs ready │  │ - Callbacks  │  │   Forest     │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────┐
│              CAMADA DE DADOS                                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Data Ingestion & Processing                         │    │
│  ├──────────┬──────────┬──────────┬──────────────────┤    │
│  │ yfinance │ CSV Raw  │ Features │ Validated        │    │
│  │ (Bolsa)  │ Output   │ Enhanced │ TimeSeries       │    │
│  └──────────┴──────────┴──────────┴──────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Pipeline de Dados (DVC Stages)

### Stage 1: Ingestão de Dados (`ingest`)

**Entrada**: Símbolo de ação (ticker), intervalo temporal  
**Processo**: Busca dados de fechamento via `yfinance`  
**Saída**: `data/raw/stock_data.csv`  
**Ferramentas**: Python, yfinance

```yaml
cmd: PYTHONPATH=. python data/ingest.py
     --ticker ${data.ticker}
     --start ${data.start} 
     --end ${data.end}
     --output ${data.output}
```

---

### Stage 2: Engenharia de Características (`feature_engineering`)

**Entrada**: Dados brutos de ações  
**Processo**:
- Normalização de valores
- Análise de Componentes Principais (PCA)
- Seleção de features
- Cálculo de Variance Inflation Factor (VIF)
- Correlação de variáveis

**Saída**: 
- `data/raw/stock_features.csv` (features processadas)
- `models/scaler_features.joblib` (scaler serializado)
- `models/pca_features.joblib` (modelo PCA)

**Métricas Geradas**:
- `reports/pca_explained_variance.csv`
- `reports/correlation_matrix.csv`
- `reports/vif_report.csv`

---

### Stage 3: Treinamento de Modelos (`train`)

**Entrada**: Features processadas, parâmetros de treinamento  
**Modelos Treinados**:

1. **PyTorch LSTM** (Deep Learning)
   - Adequado para séries temporais
   - Captura dependências de longo prazo
   - GPU-ready

2. **TensorFlow/Keras** (Deep Learning)
   - RNN/LSTM layers
   - Callbacks (EarlyStopping, checkpoints)
   - Normalização integrada

3. **Scikit-Learn Ensemble**
   - XGBoost, Random Forest
   - Baseline rápido para comparação
   - Menor footprint computacional

**Saída**:
- `modelo_pytorch.pth` (weights PyTorch)
- `modelo_keras.h5` (modelo TensorFlow)
- `modelo_sklearn.joblib` (modelo Sklearn)
- MLflow DB com logs de todos os experimentos

**Parâmetros Registrados**:
- `ticker`, `janela` (window size)
- `epochs`, `batch_size`, `learning_rate`
- `patience` (early stopping)

---

### Stage 4: Baseline & Avaliação (`baseline`)

**Entrada**: Modelos treinados, dados de teste  
**Processo**:
- Predições em dados não vistos
- Cálculo de métricas (MAE, RMSE, MAPE)
- Comparação cross-framework
- Visualizações

**Saída**:
- `reports/comparison_metrics.csv`
- `reports/predictions_vs_actual.csv`
- Gráficos comparativos

---

## 🤖 Sistema RAG (Retrieval-Augmented Generation)

### Arquitetura

```
┌─────────────────────────────────────────────────┐
│         ENTRADA: Query do Usuário              │
│  (ex: "Quais ações recomendar em 2026?")       │
└────────────────────┬────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│  1️⃣ EMBEDDING: Conversão para Vetor Semântico  │
│     SentenceTransformers: all-MiniLM-L6-v2      │
│     Dimensionalidade: 384 dimensões             │
└────────────────────┬────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│  2️⃣ RETRIEVAL: Busca Vetorial (FAISS)          │
│     Índice: L2 distance (busca euclidiana)      │
│     Top-K: 3 chunks mais similares              │
│     Tempo: O(log n) - muito eficiente           │
└────────────────────┬────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│  3️⃣ RANKING: Seleção de Contexto Relevante     │
│     Combinação: Query + Top-3 contextos         │
│     Token limit: Até 2048 tokens                │
└────────────────────┬────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│  4️⃣ GENERATION: Síntese via LLM                │
│     Modelos suportados:                         │
│     - facebook/opt-1.3b (padrão)                │
│     - distilgpt2 (rápido)                       │
│     - Fallback: Respostas simuladas             │
└────────────────────┬────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────┐
│    SAÍDA: Resposta Contextualizada em PT-BR    │
│  (ex: "Com base em análises de renda fixa...")  │
└─────────────────────────────────────────────────┘
```

### Componentes Técnicos

| Componente | Tecnologia | Função |
|-----------|-----------|--------|
| **Embedder** | SentenceTransformers | Vetorização de texto |
| **Vector Store** | FAISS | Armazenamento e busca rápida |
| **Chunking** | Custom Python | Divisão de documentos (300 tokens, overlap 50) |
| **Retriever** | Similaridade Semântica | Seleção de documentos relevantes |
| **Generator** | LLM (BentoML/vLLM) | Geração de resposta contextual |

### Fluxo de Ingestão

1. **Coleta de Dados**:
   - Web scraping de notícias financeiras (newspaper3k)
   - Upload de documentos via API `/ingest`

2. **Processamento**:
   - Chunking inteligente (300 tokens, overlap 50)
   - Limpeza e normalização de texto

3. **Vetorização**:
   - Embedding de cada chunk
   - Armazenamento em FAISS com índice L2

4. **Operacionalização**:
   - Consultas via API `/ask`
   - Respostas em tempo real

---

## 🐳 Infraestrutura & Deployment

### Docker Services

#### 1. **pipeline** (DVC Execution)
```yaml
Container: datathon-grupo-05-pipeline-1 (gerenciado pelo Docker Compose)
Image: Dockerfile
Command: dvc repro
Função: Executa pipeline completo de dados
Volume: Montagem da raiz do projeto
```

#### 2. **mlflow** (Experiment Tracking)
```yaml
Container: datathon-grupo-05-mlflow-1 (gerenciado pelo Docker Compose)
Image: Dockerfile
Porta: 5000
Command: mlflow ui --host 0.0.0.0
Função: Dashboard de experimentos e modelos
Armazenamento: SQLite (mlflow/mlflow.db)
```

#### 3. **api** (Model Serving)
```yaml
Container: datathon-grupo-05-api-1 (gerenciado pelo Docker Compose)
Image: datathon-grupo-05-api:latest
Porta: 8000
Framework: FastAPI + Uvicorn
Endpoints:
  - POST /predict (previsão com modelo selecionado)
  - POST /ask (queries RAG)
  - GET /health (status)
  - GET /models (lista modelos disponíveis)
```

#### 4. **test** (Test Runner)
```yaml
Container: datathon-grupo-05-test-1 (gerenciado pelo Docker Compose)
Image: Dockerfile.test
Command: pytest -v
Função: Validação de código e modelos
Coverage: Relatório de cobertura de testes
```

### Arquitetura de Deployment

```
┌──────────────────────────────────────────────────┐
│         Docker Compose Orquestração              │
├──────────────────────────────────────────────────┤
│                                                  │
│  ┌─────────────┐  ┌─────────────┐              │
│  │  Pipeline   │  │   MLflow    │              │
│  │  (DVC)      │  │   UI :5000  │              │
│  └─────────────┘  └─────────────┘              │
│                                                  │
│  ┌─────────────┐  ┌─────────────┐              │
│  │    API      │  │    Test     │              │
│  │  :8000      │  │   Runner    │              │
│  └─────────────┘  └─────────────┘              │
│                                                  │
│  ┌──────────────────────────────────────────┐  │
│  │    Volume Compartilhado: ./app           │  │
│  │    - Dados, modelos, logs                │  │
│  └──────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

---

## 📊 Ciclo de Vida do ML

### 1. Desenvolvimento & Experimentação

```
1. Ajustar parâmetros (params.yaml)
   ↓
2. Executar pipeline: dvc repro
   ↓
3. Modelos treinados e logados em MLflow
   ↓
4. Visualizar métricas no MLflow UI (http://localhost:5000)
   ↓
5. Comparar runs (PyTorch vs Keras vs Sklearn)
   ↓
6. Selecionar melhor modelo
```

### 2. Validação & Testes

```
1. Rodar testes unitários: docker-compose run test
   ↓
2. Validar cobertura de código
   ↓
3. Teste de integração da API
   ↓
4. Teste do RAG (offline ou online)
   ↓
5. Validar reprodutibilidade (dvc repro novamente)
```

### 3. Deployment

```
1. Build das imagens Docker
   ↓
2. Push para registry (DockerHub/ECR)
   ↓
3. Deploy em nuvem (AWS/GCP/Azure)
   ↓
4. Configurar endpoints da API
   ↓
5. Integrar com monitoring (CloudWatch/Stackdriver)
   ↓
6. A/B Testing de modelos
```

---

## 🔐 Segurança & Governança

### Camadas de Segurança

1. **Segurança de Dados**
   - Anonymização de dados sensíveis (Presidio)
   - Versionamento DVC com integridade
   - Criptografia de modelos em transit

2. **Acesso & Autenticação**
   - Módulo `security.py` em desenvolvimento
   - JWT tokens para API
   - RBAC (Role-Based Access Control)

3. **Monitoramento & Auditoria**
   - Logs estruturados
   - MLflow tracking completo
   - Rastreabilidade de previsões

---

## 📦 Estrutura de Diretórios

```
datathon-grupo-05/
├── data/
│   ├── ingest.py              # Script de ingestão (yfinance)
│   ├── raw/                   # Dados processados
│   └── features.csv           # Features engenheiradas
├── src/
│   ├── features/
│   │   └── feature_engineering.py  # PCA, normalização
│   ├── models/
│   │   ├── train.py           # Treinamento multi-framework
│   │   ├── baseline.py        # Avaliação
│   │   └── predict.py         # Predição
│   ├── rag/
│   │   ├── embedder.py        # SentenceTransformers
│   │   ├── retriever.py       # FAISS + busca
│   │   ├── generator.py       # LLM response
│   │   └── pipeline.py        # Orquestração RAG
│   ├── serving/
│   │   └── app.py             # FastAPI endpoints
│   ├── security.py            # Autenticação/Autorização
│   └── utils.py               # Helpers e logs
├── models/                    # Artefatos treinados
│   ├── *.pth (PyTorch)
│   ├── *.h5 (Keras)
│   └── *.joblib (Sklearn)
├── reports/
│   ├── pca_explained_variance.csv
│   ├── correlation_matrix.csv
│   ├── vif_report.csv
│   └── comparison_metrics.csv
├── tests/                     # Suite de testes
├── notebooks/                 # Análises exploratórias
├── docs/                      # Documentação
│   └── ARCHITECTURE.md        # Este arquivo
├── dvc.yaml                   # Pipelines DVC
├── params.yaml                # Configurações centralizadas
├── docker-compose.yaml        # Orquestração de containers
├── Dockerfile                 # Imagem principal
├── Dockerfile.test            # Imagem de testes
├── pyproject.toml             # Dependências (Poetry)
└── requirements.txt           # Compatibilidade com pip
```

---

## 🚀 Como Usar

### Setup Local

```bash
# 1. Clone e configure ambiente
cd datathon-grupo-05
python -m venv venv
venv\Scripts\activate

# 2. Instale dependências
pip install -r requirements_local.txt

# 3. Teste local (RAG offline)
python test_rag_offline.py

# 4. Execute pipeline completo
dvc repro
```

### Usando Docker

```bash
# 1. Execute pipeline
docker-compose run pipeline

# 2. Visualize experimentos
docker-compose up mlflow
# Acesse: http://localhost:5000

# 3. Inicie API
docker-compose up api
# Acesse: http://localhost:8000/docs

# 4. Rode testes
docker-compose run test
```

### Exemplos de API

```bash
# Previsão
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "PETR4",
    "model": "pytorch",
    "days_ahead": 5
  }'

# Query RAG
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Quais ações recomendar em 2026?"
  }'

# Status
curl http://localhost:8000/health
```

---

## 📈 Métricas & Monitoramento

### Métricas de Modelo

| Métrica | Fórmula | Interpretação |
|---------|---------|---------------|
| **MAE** | Média erro absoluto | Erro médio em unidades da série |
| **RMSE** | Raiz erro quadr. médio | Penaliza erros maiores |
| **MAPE** | Erro percentual absoluto médio | % de erro relativo |

### Dashboards Disponíveis

1. **MLflow UI** (localhost:5000)
   - Comparação de runs
   - Histórico de experimentos
   - Versionamento de modelos

2. **Logs & Alertas** (em desenvolvimento)
   - Monitoramento de drift
   - Alertas de performance
   - Logs estruturados

---

## 🔮 Roadmap Futuro

### Near-term (Q2-Q3 2026)
- [ ] Integração com data warehouse (BigQuery/Redshift)
- [ ] AutoML para tunagem de hiperparâmetros
- [ ] Dashboard executivo customizado
- [ ] Suporte a mais tickers

### Mid-term (Q4 2026)
- [ ] Federated learning
- [ ] Model serving com KServe/KubeFlow
- [ ] Integração com LLMs maiores (GPT-4, Claude)
- [ ] Suporte a múltiplas moedas

### Long-term (2027+)
- [ ] Sistema de recomendação multiagente
- [ ] Portfolio optimization
- [ ] Risk management integrado
- [ ] Análise em tempo real de sentimento de mercado

---

## 📞 Contato & Suporte

**Documentação**: [README.md](../README.md)  
**RAG Specifics**: [src/rag/README.md](../src/rag/README.md)  
**Testes Locais**: [TESTE_LOCAL.md](../TESTE_LOCAL.md)  
**Detalhes MLflow**: MLflow UI em http://localhost:5000

---

**Desenvolvido pelo Grupo 05 - Datathon FIAP 2026**  
*Arquitetura revisada em Maio de 2026*
