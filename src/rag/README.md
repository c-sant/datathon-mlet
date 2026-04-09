# RAG (Retrieval-Augmented Generation) System

Este documento descreve o sistema de Retrieval-Augmented Generation (RAG) implementado no projeto `datathon-grupo-05`. O RAG combina recuperação de informações relevantes de uma base de conhecimento com geração de respostas usando modelos de linguagem grandes (LLMs), permitindo respostas contextuais e precisas baseadas em dados fornecidos.

## O que é RAG?

### 🎯 Conceito Teórico

**Retrieval-Augmented Generation (RAG)** é uma arquitetura híbrida que integra **recuperação de informação** com **geração de texto**, superando limitações dos LLMs tradicionais através de acesso a conhecimento externo atualizado.

#### Problema dos LLMs Tradicionais:
- **Conhecimento estático**: Treinados em dados até uma data específica
- **Alucinações**: Podem gerar informações incorretas ou desatualizadas
- **Falta de transparência**: Difícil rastrear fonte das informações

#### Solução RAG:
- **Conhecimento dinâmico**: Base de dados externa consultada em tempo real
- **Respostas fundamentadas**: Baseadas em contexto recuperado verificável
- **Transparência**: Possibilita citar fontes e justificar respostas

### ⚙️ Arquitetura Técnica

```
Query → Embedding → Retrieval → Re-ranking → Generation → Response
```

#### Componentes Técnicos:
- **Embedder**: Transforma texto em vetores densos (SentenceTransformers)
- **Vector Store**: Índice de busca eficiente (FAISS - L2 distance)
- **Retriever**: Busca semântica por similaridade vetorial
- **Re-ranker**: Reordenação opcional dos resultados (não implementado)
- **Generator**: LLM condicionado no contexto recuperado

#### Fluxo de Dados:
1. **Indexação**: Documentos → Chunks → Embeddings → FAISS
2. **Query**: Texto → Embedding → Top-K similares → Contexto
3. **Geração**: Query + Contexto → LLM → Resposta final

#### Métricas de Qualidade:
- **Relevância**: Precisão na recuperação de chunks pertinentes
- **Coerência**: Qualidade da resposta gerada
- **Fidelidade**: Aderência ao contexto fornecido
- **Latência**: Tempo de resposta end-to-end

### 🎨 Vantagens Técnicas do RAG

- **Atualização Contínua**: Base de conhecimento pode ser atualizada sem retreinar o modelo
- **Explicabilidade**: Respostas podem ser rastreadas até fontes específicas
- **Redução de Alucinações**: Contexto externo reduz geração de informações incorretas
- **Escalabilidade**: Separação entre recuperação e geração permite otimização independente
- **Custo-Efetividade**: Não requer fine-tuning completo do LLM para novos domínios

## Visão Geral

O sistema RAG permite:
- **Ingestão dinâmica de documentos**: Adicione textos personalizados à base de conhecimento via API.
- **Recuperação contextual**: Busca os trechos mais relevantes para uma consulta usando embeddings e FAISS.
- **Geração de respostas**: Produz respostas em português usando um LLM (via serviço BentoML ou fallback local).
- **Monitoramento**: Integração com MLflow para rastreamento de experimentos e logs.

### Arquitetura

```
[Usuário] → [API FastAPI] → [Retriever] → [Base de Conhecimento (FAISS)]
                     ↓
              [Generator (BentoML/vLLM)] → [Resposta]
```

- **Data Loader**: Carrega documentos iniciais (notícias financeiras).
- **Ingest**: Processa documentos, gera embeddings e indexa no FAISS.
- **Retriever**: Busca chunks relevantes para uma query.
- **Generator**: Gera resposta baseada no contexto recuperado.
- **API**: Expõe endpoints para ingestão e consulta.

## Componentes

### 1. `data_loader.py`
- **Função**: Carrega documentos iniciais de fontes externas (notícias via `newspaper3k`).
- **URLs padrão**:
  - https://www.seudinheiro.com/mercados
  - https://einvestidor.estadao.com.br/mercado
  - https://www.infomoney.com.br/mercados/
- **Saída**: Lista de dicionários com `id`, `title`, `text`.

### 2. `ingest.py`
- **Função**: Processa documentos, divide em chunks, gera embeddings e indexa no FAISS.
- **Modelo de embeddings**: `all-MiniLM-L6-v2` (SentenceTransformers).
- **Chunking**: Tamanho 300 palavras, overlap 50.
- **Função principal**: `ingest_documents(documents, overwrite=True)`
  - Aceita lista de documentos (`id`, `title`, `text`).
  - Reconstrói o índice FAISS dinamicamente.
- **Logs**: Integra com MLflow para rastreamento.

### 3. `retriever.py`
- **Função**: Busca os top-k chunks mais relevantes para uma query.
- **Processo**:
  1. Gera embedding da query.
  2. Busca no índice FAISS (distância L2).
  3. Retorna chunks com metadados (rank, distância, texto, doc_id, etc.).
- **Parâmetros**: `top_k` (padrão 3).

### 4. `generator.py`
- **Função**: Gera resposta baseada na query e contexto.
- **Serviço primário**: BentoML (URL: `http://localhost:3000/generate`).
  - Modelo: `facebook/opt-1.3b`.
  - Prompt: "Responda em português.\nPergunta: {query}\nContexto: {context}\nResposta:"
- **Fallback**: Transformers pipeline local (`facebook/opt-1.3b`).
- **Configuração**: Use `RAG_GENERATOR_URL` para sobrescrever URL; `USE_BENTO_GENERATOR=false` para forçar fallback.

### 5. `api.py`
- **Framework**: FastAPI.
- **Endpoints**:
  - `POST /ingest`: Ingesta documentos.
    - Payload: `{"docs": [{"id": "...", "title": "...", "text": "..."}], "overwrite": true}`
    - Resposta: Estatísticas da ingestão.
  - `GET /query`: Consulta RAG.
    - Parâmetros: `q` (query), `top_k` (opcional, padrão 3).
    - Resposta: `{"query": "...", "top_k": 3, "context": "...", "answer": "..."}`
- **Validações**: Erro 503 se índice não disponível; erro 400 para dados inválidos.

## Dependências

- `fastapi` + `uvicorn`: API web.
- `sentence-transformers`: Embeddings.
- `faiss-cpu`: Busca vetorial.
- `transformers`: Geração local.
- `requests`: Chamadas para BentoML.
- `mlflow`: Logs e experimentos.
- `newspaper3k`: Carregamento de notícias.
- `pydantic`: Validação de dados.

Instale via `pip install -r requirements_local.txt -f https://download.pytorch.org/whl/torch_stable.html` a partir da raiz do projeto.

## Como Executar

### 1. Preparar Ambiente
```bash
cd datathon-grupo-05
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate no Windows
pip install -r requirements_local.txt -f https://download.pytorch.org/whl/torch_stable.html
```

### 2. Iniciar Serviço Bento (Opcional, Recomendado)
```bash
cd generator/serving
bentoml serve app:svc --port 3000
```

### 3. Executar API RAG
```bash
cd src/rag
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Testar Localmente
```bash
python run_local.py
```

## Exemplos de Uso

### Ingestão de Documentos
```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "docs": [
      {"id": "doc1", "title": "Ações em 2026", "text": "Texto sobre recomendações de ações para 2026..."},
      {"id": "doc2", "title": "Mercado Brasileiro", "text": "Análise do mercado brasileiro..."}
    ],
    "overwrite": true
  }'
```

Resposta:
```json
{
  "status": "ok",
  "doc_count": 2,
  "chunk_count": 5,
  "overwrite": true
}
```

### Consulta RAG
```bash
curl "http://localhost:8000/query?q=Quais%20ações%20recomendadas%20para%202026?&top_k=3"
```

Resposta:
```json
{
  "query": "Quais ações recomendadas para 2026?",
  "top_k": 3,
  "context": "Texto relevante do contexto recuperado...",
  "answer": "Resposta gerada pelo LLM baseada no contexto."
}
```

## Monitoramento e Logs

- **MLflow**: Runs automáticos em `ingest.py` (experimento "RAG_ingest").
- **Métricas**: Número de documentos/chunks, modelo de embeddings, vector store.
- **Artefatos**: Código fonte logado.

Acesse MLflow UI: `mlflow ui` (porta padrão 5000).

## Próximos Passos

- Integrar com pipeline CI/CD para deploy automático.
- Adicionar autenticação/autorização aos endpoints.
- Melhorar chunking (ex.: por sentenças ou seções).
- Suporte a múltiplas línguas ou modelos de embeddings.
- Dashboard para visualização de queries e contexto.

## Contribuição

1. Faça fork do repositório.
2. Crie uma branch para sua feature.
3. Implemente mudanças e adicione testes.
4. Abra um Pull Request.

Para questões, abra uma issue no GitHub.

---

**Última atualização**: Abril 2026
**Versão**: 1.0</content>
<parameter name="filePath">c:\Users\cabri\Documents\Fiap\challenge5\datathon-grupo-05\src\rag\README.md