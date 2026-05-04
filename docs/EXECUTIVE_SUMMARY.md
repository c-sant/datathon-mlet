# 📊 DATATHON MLET
## Sumário Executivo - Visão de Negócios

**Data de Apresentação:** Maio 2026  
**Grupo:** Grupo 05 - FIAP  
**Status:** Sistema em Produção  

---

## 🎯 O QUE É O DATATHON MLET?

Um **sistema inteligente de previsão de ações em bolsa de valores** que combina:

✅ **Machine Learning Avançado** - 3 modelos treinados (PyTorch, TensorFlow, Scikit-Learn)  
✅ **Inteligência Artificial Conversacional** - RAG para análise contextualizada  
✅ **Automação Completa** - Pipeline reprodutível com DVC  
✅ **Rastreabilidade Total** - Todos os experimentos registrados em MLflow  

---

## 💼 VALOR DE NEGÓCIO

| Benefício | Impacto |
|-----------|---------|
| **Previsões Precisas** | Reduz erro de predição com 3 modelos complementares |
| **Análise Contextual** | Sistema RAG fornece recomendações fundamentadas em dados |
| **Escalabilidade** | Arquitetura containerizada, pronta para nuvem |
| **Reprodutibilidade** | Todo experimento pode ser replicado e auditado |
| **Decisões Informadas** | Dashboard em tempo real para executivos e analistas |

---

## 🏗️ ARQUITETURA EM 3 CAMADAS

### 1️⃣ **CAMADA DE DADOS**
```
Bolsa de Valores (yfinance)
        ↓
Extração & Normalização
        ↓
Engenharia de Features (PCA)
        ↓
Dados Prontos para Modelagem
```

### 2️⃣ **CAMADA DE INTELIGÊNCIA**
```
3 MODELOS TREINADOS:
├─ 🔷 PyTorch LSTM (Séries Temporais)
├─ 🔵 TensorFlow/Keras (Deep Learning)
└─ 🟢 Scikit-Learn (Ensemble Rápido)

+ SISTEMA RAG (IA Conversacional)
├─ Busca em Base de Conhecimento
├─ Respostas Contextualizadas
└─ Análise em Linguagem Natural
```

### 3️⃣ **CAMADA DE SAÍDA**
```
API REST (Porta 8000)
├─ Previsões de Preços
├─ Análise Contextual (IA)
└─ Health Checks

Dashboard MLflow (Porta 5000)
├─ Comparação de Modelos
├─ Métricas de Desempenho
└─ Histórico de Experimentos
```

---

## 🚀 FUNCIONALIDADES PRINCIPAIS

### 1. Previsão de Ações
```
POST /predict
{
  "ticker": "PETR4",
  "model": "pytorch",
  "days_ahead": 5
}

Resposta:
{
  "predictions": [29.50, 29.65, 29.80, ...],
  "confidence": 0.87,
  "model": "pytorch_lstm"
}
```

### 2. Análise Inteligente (RAG)
```
POST /ask
{
  "query": "Quais ações recomendar em 2026?"
}

Resposta:
{
  "answer": "Com base em análises de renda fixa 
             e juros em alta...",
  "context_sources": 3,
  "confidence": 0.92
}
```

### 3. Monitoramento de Experimentos
```
MLflow UI: http://localhost:5000
- Todas as métricas: MAE, RMSE, MAPE
- Versionamento de modelos
- Histórico de execuções
- Comparação cross-framework
```

---

## 📈 MÉTRICAS DE DESEMPENHO

| Métrica | Descrição | Valor Esperado |
|---------|-----------|----------------|
| **MAE** | Erro Médio Absoluto | < 2% do preço |
| **RMSE** | Raiz do Erro Quadrático | Penaliza outliers |
| **MAPE** | Erro Percentual Absoluto | < 5% em média |

**Benchmark**: Comparação de 3 frameworks para validação robusta

---

## 🐳 INFRAESTRUTURA

### Containerização Completa
```
docker-compose up                    # Inicia todos os serviços
├─ pipeline    → Executa DVC repro
├─ mlflow      → Dashboard UI
├─ api         → Serving FastAPI
└─ test        → Validação automatizada
```

### Deployment
```
✅ Desenvolvido com Python 3.13
✅ Isolamento total com Docker
✅ Pronto para: AWS, GCP, Azure
✅ Suporte a GPU via PyTorch
```

---

## 🤖 SISTEMA RAG - Geração Aumentada por Recuperação

### Como Funciona

```
Pergunta:
"Quais ações recomendar em 2026?"
        ↓
1. EMBEDDING: Converte pergunta em vetor
        ↓
2. RETRIEVAL: Busca documentos similares (FAISS)
        ↓
3. RANKING: Seleciona contexto relevante
        ↓
4. GENERATION: IA gera resposta contextualizada
        ↓
Resposta:
"Com base em análises de renda fixa..."
```

### Benefícios
- ✅ Respostas baseadas em dados reais
- ✅ Reduz "alucinações" de IA
- ✅ Explicabilidade completa
- ✅ Atualização em tempo real

---

## 📊 PIPELINE DE DADOS (DVC)

### Estágios Automatizados

**1. INGEST** → Dados do Yahoo Finance  
**2. FEATURE ENGINEERING** → PCA, Normalização  
**3. TRAIN** → 3 modelos simultâneos  
**4. BASELINE** → Avaliação e comparação  

### Reprodutibilidade
```
dvc repro
# Executa pipeline completo
# Todos os dados versionados
# Todos os modelos rastreáveis
# Auditoria total possível
```

---

## 💡 CASOS DE USO

### Para Traders
- ✅ Previsões de curto prazo
- ✅ Análise contextual de mercado
- ✅ Alertas automáticos

### Para Analistas
- ✅ Comparação de estratégias
- ✅ Backtesting automático
- ✅ Relatórios estruturados

### Para Executivos
- ✅ Dashboard consolidado
- ✅ KPIs em tempo real
- ✅ ROI por modelo

---

## 🔐 SEGURANÇA & CONFORMIDADE

- ✅ Anonymização de dados sensíveis (Presidio)
- ✅ Rastreabilidade completa (MLflow)
- ✅ Versionamento seguro (DVC)
- ✅ Logs estruturados para auditoria
- ✅ HTTPS para endpoints

---

## 📅 ROADMAP

### Q2-Q3 2026 (Próximos)
- [ ] Integração com data warehouse
- [ ] Dashboard executivo customizado
- [ ] Suporte a mais tickers

### Q4 2026
- [ ] AutoML para otimização automática
- [ ] Integração com LLMs maiores
- [ ] Risk management integrado

### 2027+
- [ ] Portfolio optimization
- [ ] Análise de sentimento em tempo real
- [ ] Sistema multiagente

---

## 🎓 MATURITY MODEL - Microsoft MLOps

Nosso projeto está nos **Níveis 2-3**:

| Nível | Status | Detalhes |
|-------|--------|----------|
| **Nível 1** | ❌ | Superado |
| **Nível 2** | ✅ | Métricas registradas, Parametrização |
| **Nível 3** | ✅ | MLflow integrado, Benchmark automatizado |
| **Nível 4** | ⚠️ | Em planejamento (AutoML) |
| **Nível 5** | ⚠️ | Em planejamento (Otimização total) |

---

## 🎯 PRÓXIMOS PASSOS

1. **Imediato**
   - [ ] Deploy em staging
   - [ ] Validação com dados reais
   - [ ] Treinamento de equipe

2. **Curto Prazo**
   - [ ] Setup em nuvem (AWS/GCP)
   - [ ] Integração com sistemas existentes
   - [ ] Ajuste fino de modelos

3. **Longo Prazo**
   - [ ] Escalabilidade para múltiplas moedas
   - [ ] IA conversacional avançada
   - [ ] Automação de recomendações

---

## 📞 SUPORTE & DOCUMENTAÇÃO

- **Arquitetura Técnica**: [docs/ARCHITECTURE.md](ARCHITECTURE.md)
- **RAG Detalhado**: [src/rag/README.md](../src/rag/README.md)
- **Testes Locais**: [TESTE_LOCAL.md](../TESTE_LOCAL.md)
- **Dashboard MLflow**: http://localhost:5000
- **API Docs**: http://localhost:8000/docs

---

## ✨ CONCLUSÃO

O **Datathon MLet** é uma solução moderna, escalável e totalmente rastreável para previsão de ações com IA. Combina o melhor de:

- 🤖 **Machine Learning**: 3 frameworks complementares
- 🧠 **Inteligência Artificial**: Sistema RAG contextualizado
- 🔄 **DevOps**: Pipeline reprodutível e automatizado
- 📊 **MLOps**: Rastreabilidade total e governança
- 🐳 **Cloud-Native**: Containerizado e pronto para escala

**Pronto para transformar decisões de investimento com dados e IA.**

---

**Desenvolvido pelo Grupo 05 - Datathon FIAP**  
*Maio 2026*
