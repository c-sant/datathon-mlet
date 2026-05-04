# 📚 Documentação de Arquitetura - Guia de Navegação

## 📋 Resumo

Este diretório contém a documentação completa de arquitetura do projeto **Datathon MLet**, preparada para diferentes públicos:

- **👨‍💻 Equipe Técnica**: Documentação detalhada
- **📈 Executivos/Gestores**: Sumário executivo com foco em negócios
- **🎨 Visual**: Diagramas de arquitetura para apresentações

---

## 📂 Arquivos Disponíveis

### 1. **ARCHITECTURE.md** 📘
**Para**: Arquitetos de software, Engenheiros, Tech Leads  
**Conteúdo**:
- Visão completa em camadas
- Pipeline DVC detalhado (4 estágios)
- Sistema RAG com especificações técnicas
- Infraestrutura Docker
- Estrutura de diretórios
- Exemplos de API
- Métricas técnicas

**Quando usar**:
- Integração técnica com sistemas existentes
- Desenvolvimento de novas features
- Troubleshooting de problemas
- Documentação para repositório

---

### 2. **EXECUTIVE_SUMMARY.md** 📊
**Para**: Executivos, Gestores, Stakeholders  
**Conteúdo**:
- Sumário executivo conciso
- Valor de negócio em 3 camadas
- Funcionalidades principais
- Casos de uso reais
- Roadmap de negócio
- Próximos passos

**Quando usar**:
- Apresentações de negócio
- Relatórios para C-Level
- Aprovações de investimento
- Comunicação com stakeholders

---

## 🎨 Diagramas Renderizados

### Diagrama 1: **Arquitetura Completa do Sistema**
```
├─ Camada de Apresentação (API FastAPI)
├─ Camada de Monitoramento (MLflow, DVC)
├─ Camada de Aplicação (RAG, Feature Engineering)
├─ Camada de Modelos (PyTorch, Keras, Sklearn)
├─ Camada de Dados (Ingestão, Raw, Features)
├─ Infraestrutura Docker (4 containers)
└─ Artefatos & Versioning (Modelos, Preprocessors, Reports)
```
**Uso**: Visão técnica completa, documentação interna

---

### Diagrama 2: **Fluxo Simplificado - Visão Executiva**
```
Entrada → Processamento → Inteligência → Saída → Usuários
```
**Uso**: Apresentações para negócios, comunicação executiva

---

### Diagrama 3: **Fluxo RAG - Sistema de IA**
```
Query → Embedding → Retrieval → Ranking → Generation → Resposta
```
**Uso**: Explicar sistema de IA para stakeholders

---

### Diagrama 4: **Ciclo de Vida MLOps**
```
Dev → Experimentação → Validação → Deployment → Produção → Observabilidade
```
**Uso**: Explicar processo de desenvolvimento, testes e deployment

---

## 🎯 Como Usar Cada Documento

### Cenário 1: Apresentação para Executivos (30 min)
1. Abra **EXECUTIVE_SUMMARY.md**
2. Use **Diagrama 2** (Fluxo Simplificado) para visão geral
3. Use **Diagrama 3** (RAG) para explicar IA
4. Destaque seção "Valor de Negócio"
5. Mencione "Roadmap"

---

### Cenário 2: Onboarding Técnico (2 horas)
1. Leia **ARCHITECTURE.md** completo
2. Estude **Diagrama 1** em detalhes
3. Execute localmente: `docker-compose up`
4. Explore MLflow UI: `http://localhost:5000`
5. Teste API: `http://localhost:8000/docs`

---

### Cenário 3: Integração com Outro Sistema (4 horas)
1. Estude **ARCHITECTURE.md** seção "API"
2. Revise **Diagrama 1** camada de apresentação
3. Leia exemplos de chamadas em "Como Usar"
4. Execute testes de integração
5. Documente em arquivo de integração

---

### Cenário 4: Troubleshooting/Debug (1 hora)
1. Consulte **ARCHITECTURE.md** seção relevante
2. Verifique **Diagrama 4** para entender ciclo
3. Consulte logs em `mlflow/mlflow.db`
4. Execute: `docker-compose run test`

---

## 📊 Estatísticas do Projeto

| Aspecto | Detalhe |
|---------|---------|
| **Camadas de Arquitetura** | 7 camadas |
| **Modelos ML** | 3 frameworks |
| **Containers Docker** | 4 serviços |
| **Estágios DVC** | 4 pipelines |
| **Endpoints API** | 4+ endpoints |
| **Cobertura MLOps** | Níveis 2-3 (Microsoft) |
| **Base de Conhecimento** | FAISS + SentenceTransformers |
| **Linguagem Padrão** | Python 3.13 |

---

## 🔗 Referências Rápidas

### Documentação Técnica Interna
- [RAG Específico](../src/rag/README.md)
- [Testes Locais](../TESTE_LOCAL.md)
- [README Principal](../README.md)

### Endpoints da API
```
POST   /predict    → Previsão de ações
POST   /ask        → Query RAG
GET    /health     → Status do sistema
GET    /models     → Modelos disponíveis
```

### Portas
- **5000**: MLflow UI
- **8000**: API FastAPI
- **8501**: Streamlit (se habilitado)

### Comandos Úteis
```bash
# Executar pipeline completo
dvc repro

# Iniciar todos os serviços
docker-compose up

# Visualizar experimentos
docker-compose up mlflow

# Testes
docker-compose run test

# API Documentation
http://localhost:8000/docs
```

---

## 🚀 Próximos Passos

### Para Desenvolvimento
- [ ] Revisar [ARCHITECTURE.md](ARCHITECTURE.md)
- [ ] Executar localmente com Docker
- [ ] Estudar estágios DVC em `dvc.yaml`
- [ ] Explorar código em `src/`

### Para Apresentação
- [ ] Preparar slides com diagramas
- [ ] Usar [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md)
- [ ] Preparar demo ao vivo
- [ ] Documentar Q&A

### Para Integração
- [ ] Mapear dependências em `pyproject.toml`
- [ ] Revisar endpoints da API
- [ ] Preparar plano de integração
- [ ] Agendar reunião técnica

---

## 📞 Suporte

**Questões Técnicas**: Consulte [ARCHITECTURE.md](ARCHITECTURE.md)  
**Questões de Negócio**: Consulte [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md)  
**Problemas de Setup**: Veja [../TESTE_LOCAL.md](../TESTE_LOCAL.md)  
**Código Fonte**: Explore [../src/](../src/)

---

## 📝 Versão & Histórico

| Versão | Data | Mudanças |
|--------|------|----------|
| 1.0 | Maio 2026 | Criação inicial |

---

**Desenvolvido pelo Grupo 05 - Datathon FIAP**  
*Documentação completa da arquitetura do sistema MLet para previsão de ações*

---

## 🎓 Notas Adicionais

### Por que 3 modelos?
- **Diversidade**: Diferentes arquiteturas capturam diferentes padrões
- **Robustez**: Se um falha, os outros fornecem previsões
- **Benchmark**: Comparação automática de desempenho
- **Confiança**: Consensus melhora predição

### Por que RAG?
- **Atualização**: Sem retraining, conhecimento é atualizado
- **Explicabilidade**: Respostas têm contexto verificável
- **Reduz Alucinações**: IA não "inventa" informações
- **Velocidade**: Respostas imediatas, sem latência de treinamento

### Por que DVC + MLflow?
- **DVC**: Versionamento de dados e reprodutibilidade
- **MLflow**: Tracking de experimentos e governance
- **Juntos**: Pipeline completo rastreável e auditável

---

**Última atualização**: Maio 2026
