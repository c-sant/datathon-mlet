# System Card — Datathon MLET (Grupo 05)

Documento que descreve o **sistema completo** colocado em operação: o que faz, como funciona, em que contextos pode ser usado com responsabilidade e quais riscos foram avaliados. Inspirado em System Cards publicados por OpenAI e Anthropic e nas diretrizes do **NIST AI Risk Management Framework**.

> **Status do documento.** v1.1 — 2026-05-03
> **Versão do sistema descrita.** Branch `dev` (snapshot atual do repositório, com vLLM remoto em RunPod e stack de observabilidade Prometheus/Grafana/Langfuse).
> **Dono do sistema.** Grupo 05 — FIAP MLET, Fase Datathon.
> **Encarregado (DPO).** *A designar* — ver [docs/LGPD_PLAN.md](docs/LGPD_PLAN.md#8-encarregado-dpo--art-41).

---

## Sumário

1. [Visão geral do sistema](#1-visão-geral-do-sistema)
2. [Casos de uso pretendidos](#2-casos-de-uso-pretendidos)
3. [Casos de uso fora do escopo](#3-casos-de-uso-fora-do-escopo)
4. [Arquitetura](#4-arquitetura)
5. [Componentes](#5-componentes)
6. [Dados](#6-dados)
7. [Avaliação de capacidades](#7-avaliação-de-capacidades)
8. [Avaliação de segurança](#8-avaliação-de-segurança)
9. [Avaliação ética, fairness e explicabilidade](#9-avaliação-ética-fairness-e-explicabilidade)
10. [Limitações e riscos conhecidos](#10-limitações-e-riscos-conhecidos)
11. [Conformidade regulatória](#11-conformidade-regulatória)
12. [Monitoramento e atualização](#12-monitoramento-e-atualização)
13. [Histórico de versões](#13-histórico-de-versões)

---

## 1. Visão geral do sistema

**Nome.** Sistema de Análise Financeira Assistida por IA — Datathon MLET / Grupo 05.

**Propósito.** Combinar (a) **previsão estatística** de preços de ações em horizonte curto via modelos de regressão e (b) **análise contextual** sobre documentos financeiros (notícias, boletins, relatórios) via RAG e agente ReAct, expostos em uma API HTTP.

**Modalidades de saída.**
- Previsão numérica de preço de fechamento (regressão).
- Resposta em linguagem natural a perguntas sobre o conteúdo da base de conhecimento.
- Recomendação categórica de alocação (gerada por LLM ou por respostas simuladas — ver §10).

**Status.** Projeto acadêmico em estágio de **prova de conceito**. Sem audiência pública, sem operação financeira real, sem certificação CVM.

---

## 2. Casos de uso pretendidos

| Caso de uso | Audiência | Componente principal |
|-------------|-----------|----------------------|
| Demonstração acadêmica de pipeline MLOps (treino + tracking + deploy + observabilidade) | Avaliadores FIAP, banca | [src/models/train.py](src/models/train.py), MLflow, [dvc.yaml](dvc.yaml), [docker-compose.yaml](docker-compose.yaml) |
| Comparação de frameworks (PyTorch vs Sklearn vs Keras) sobre série multivariada | Estudantes, professores | [src/models/train.py](src/models/train.py), [src/features/feature_engineering.py](src/features/feature_engineering.py) |
| Exploração de RAG com FAISS sobre corpus financeiro pequeno + métricas MLflow | Estudantes | [src/rag/](src/rag/), [src/rag/mlflow_loader.py](src/rag/mlflow_loader.py) |
| Exploração de agente ReAct com chamada de ferramentas e LLM remoto | Estudantes | [src/agent/react_agent.py](src/agent/react_agent.py), endpoint vLLM em RunPod |
| Demonstração de stack de observabilidade para sistemas de IA | Estudantes, banca | Prometheus + Grafana + Langfuse ([docker-compose.yaml:96-172](docker-compose.yaml#L96-L172)) |

---

## 3. Casos de uso fora do escopo

Os usos abaixo são **explicitamente desencorajados** e o sistema **não** está apto a sustentá-los:

- ❌ **Recomendação real de investimento a clientes** (atividade regulada — Resolução CVM 39/2021).
- ❌ **Trading automatizado** ou execução de ordens com base nas previsões.
- ❌ **Aconselhamento financeiro a leigos** sem disclaimer e sem revisão humana.
- ❌ **Análise de ativos com baixa liquidez** (small caps com histórico curto) — modelo não foi validado nesse regime (ver [docs/EXPLAINABILITY_FAIRNESS.md §5.2 R1](docs/EXPLAINABILITY_FAIRNESS.md#52-riscos-concretos-no-projeto)).
- ❌ **Decisão crítica não revisada por pessoa natural** — viola Art. 20 da LGPD.
- ❌ **Tratamento de dados pessoais** sem o estado de adequação descrito em [docs/LGPD_PLAN.md](docs/LGPD_PLAN.md).
- ❌ **Operação em produção pública** no estado atual — ver §8 (sem autenticação, CORS aberto, sem rate limiting).

---

## 4. Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                       Cliente HTTP                              │
│                  (curl / docs/index.html)                       │
└─────┬────────────────┬────────────────────────────┬─────────────┘
      │                │                            │
      ▼                ▼                            ▼
┌───────────┐  ┌─────────────────┐    ┌──────────────────────┐
│ /ingest   │  │ /ingest_mlflow  │    │  /query   /agent     │
│ (docs)    │  │ (métricas DVC)  │    │  (RAG + Agente)      │
└─────┬─────┘  └─────┬───────────┘    └────┬─────────────────┘
      │              │                     │
      ▼              ▼                     ▼
┌──────────────────────────┐    ┌─────────────────────┐
│ embedder                 │    │ Agente ReAct        │
│ (SentenceTransformer     │    │ (react_agent.py)    │
│  all-MiniLM-L6-v2)       │    │ ┌─────────────────┐ │
└────────┬─────────────────┘    │ │ search_docs     │ │
         │                      │ │ fetch_news      │ │
         ▼                      │ │ summarize       │ │
┌──────────────────────────┐    │ └─────────────────┘ │
│ FAISS IndexFlatL2        │◄───┤                     │
│ (in-memory, global)      │    └────┬────────────────┘
└──────────────────────────┘         │
                                     ▼
                         ┌─────────────────────────┐
                         │ Generator               │
                         │ (vLLM remoto na RunPod  │
                         │  via VLLM_BASE_URL,     │
                         │  fallback HF local      │
                         │  ou modo simulated)     │
                         └─────────────────────────┘

┌──────────────────────────────────────────┐  ┌──────────────────────────┐
│ Modelos preditivos (offline / batch)     │  │ Observabilidade          │
│ MLP PyTorch · MLP Sklearn · LSTM Keras   │  │ Prometheus  → Grafana    │
│ Pipeline DVC + tracking MLflow           │  │ Langfuse (LLM tracing)   │
│ Métricas reingeridas via /ingest_mlflow  │  │ MLflow UI                │
└──────────────────────────────────────────┘  └──────────────────────────┘
```

Detalhamento: [src/serving/app.py](src/serving/app.py), [src/agent/react_agent.py](src/agent/react_agent.py), [src/rag/](src/rag/), [docker-compose.yaml](docker-compose.yaml).

---

## 5. Componentes

### 5.1 API HTTP (FastAPI)
| Atributo | Valor |
|----------|-------|
| Arquivo | [src/serving/app.py](src/serving/app.py) |
| Endpoints | `POST /ingest`, `POST /ingest_mlflow`, `GET /query`, `POST /agent` |
| Porta | `:8000` ([docker-compose.yaml:52-54](docker-compose.yaml#L52-L54)) |
| Autenticação | **Nenhuma** (gap conhecido — ver [docs/OWASP.md §3.5](docs/OWASP.md#35-api12023--a052021--broken-access-control--security-misconfiguration)) |
| CORS | `allow_origins=["*"]` + `allow_credentials=True` (gap — ver [docs/OWASP.md §3.5](docs/OWASP.md#35-api12023--a052021--broken-access-control--security-misconfiguration)) |

### 5.2 Modelos preditivos
Documentados em detalhe no [Model Card](docs/MODEL_CARD.md).

| Modelo | Framework | Artefato |
|--------|-----------|----------|
| MLP_PyTorch | PyTorch 2.x | `modelo_pytorch.pth` |
| MLPRegressor | Scikit-learn | `modelo_sklearn.joblib` |
| LSTM | Keras / TensorFlow | `modelo_{ticker}.keras` |
| Baseline | Persistence (último valor) | computado em runtime |

### 5.3 Pipeline RAG
| Componente | Implementação | Arquivo |
|------------|---------------|---------|
| Embedder | `SentenceTransformer("all-MiniLM-L6-v2")` | [src/rag/embedding.py:32-45](src/rag/embedding.py#L32-L45) |
| Chunker | Janela 300 palavras, overlap 50 | [src/rag/embedding.py:13-22](src/rag/embedding.py#L13-L22) |
| Vector Store | `faiss.IndexFlatL2` (in-memory, global) | [src/rag/embedding.py:79-82](src/rag/embedding.py#L79-L82) |
| Retriever | Top-K por distância L2 | [src/rag/retriever.py](src/rag/retriever.py) |
| Generator | vLLM remoto (`qwen2.5-0.5b-awq` em RunPod) via `VLLM_BASE_URL`, fallback BentoML ou HF local (`flan-t5-base`) ou `simulated` | [src/rag/generator.py](src/rag/generator.py) |
| Loader MLflow | Lê métricas/parâmetros do pipeline DVC e gera documentos para `/ingest_mlflow` | [src/rag/mlflow_loader.py](src/rag/mlflow_loader.py) |

### 5.4 Agente ReAct
| Atributo | Valor |
|----------|-------|
| Arquivo | [src/agent/react_agent.py](src/agent/react_agent.py) |
| Ferramentas | `search_documents`, `fetch_news`, `summarize_context` |
| `max_steps` | Configurável via [config/](config/) |
| Trace exposto | Sim (campo `trace` na resposta) |

### 5.5 Infraestrutura
| Item | Detalhe |
|------|---------|
| Tracking de experimentos | MLflow (`sqlite:///mlflow/mlflow.db`) — UI em `:5000` ([docker-compose.yaml:15-30](docker-compose.yaml#L15-L30)) |
| Versionamento de dados | DVC ([dvc.yaml](dvc.yaml), [params.yaml](params.yaml)) |
| Containerização | Docker Compose ([docker-compose.yaml](docker-compose.yaml)) |
| Inferência LLM remota | vLLM com modelo AWQ (`qwen2.5-0.5b-awq`) hospedado na RunPod — `VLLM_BASE_URL` ([generator.py:12](src/rag/generator.py#L12)) |
| Métricas operacionais | Prometheus em `:9090` ([docker-compose.yaml:96-113](docker-compose.yaml#L96-L113)) |
| Dashboards | Grafana em `:3001` ([docker-compose.yaml:115-135](docker-compose.yaml#L115-L135)) |
| Observabilidade de LLM | Langfuse em `:3000` + PostgreSQL ([docker-compose.yaml:138-172](docker-compose.yaml#L138-L172)) |
| Gerenciamento de deps | Poetry ([pyproject.toml](pyproject.toml)) |

---

## 6. Dados

### 6.1 Dados de treino dos modelos preditivos
- **Fonte.** CSV de features tratadas em `data/raw/stock_features.csv` — gerado por [src/features/feature_engineering.py](src/features/feature_engineering.py) a partir de `data/raw/stock_data.csv`.
- **Janela.** Configurável via [params.yaml](params.yaml) e [config/model_config.yaml](config/model_config.yaml).
- **Split.** Temporal 80/20 — tag `split_type=temporal_80_20` em [train.py:317](src/models/train.py#L317).
- **Pré-processamento.** `MinMaxScaler` no target em `preparar_series_features` ([train.py:154-188](src/models/train.py#L154-L188)).

### 6.2 Dados ingeridos no RAG
- **Fonte primária.** Notícias carregadas via [src/rag/data_loader.py](src/rag/data_loader.py) (sem whitelist explícita — gap em [docs/OWASP.md §3.3](docs/OWASP.md#33-llm042025--data-and-model-poisoning)).
- **Fonte secundária.** Documentos enviados pelo cliente em `POST /ingest`.
- **Fonte terciária.** Métricas e parâmetros de runs MLflow injetados via `POST /ingest_mlflow` ([src/rag/mlflow_loader.py](src/rag/mlflow_loader.py)).
- **Persistência.** Apenas em memória do processo (FAISS in-memory). **Não há** banco persistente para o índice.

### 6.3 Dados pessoais
Inventário completo em [docs/LGPD_PLAN.md §2](docs/LGPD_PLAN.md#2-mapeamento-de-dados-pessoais-registro-de-operações-de-tratamento).

---

## 7. Avaliação de capacidades

### 7.1 Modelos preditivos

Métricas de teste registradas no MLflow pelo pipeline DVC ([dvc.yaml](dvc.yaml)):

| Modelo | MAE ↓ | RMSE ↓ | MAPE ↓ |
|--------|-------|--------|--------|
| Baseline (persistence) | `mae_baseline` | `rmse_baseline` | `mape_baseline` |
| MLP PyTorch | `mae_pytorch` | `rmse_pytorch` | `mape_pytorch` |
| MLP Sklearn | `mae_sklearn` | `rmse_sklearn` | `mape_sklearn` |
| LSTM Keras | `mae_keras` | `rmse_keras` | `mape_keras` |

> Os valores numéricos são lidos do MLflow do último run e expostos no Grafana via dashboard. O System Card **não** os congela porque variam por ticker e janela; consulte o run mais recente, o [Model Card](docs/MODEL_CARD.md) ou consulte via `POST /ingest_mlflow` + `GET /query?q=compare os modelos`.

### 7.2 RAG

Avaliação ainda **não automatizada**. Plano em [docs/EXPLAINABILITY_FAIRNESS.md §6 Sprint 3](docs/EXPLAINABILITY_FAIRNESS.md#6-plano-de-implementação):
- Faithfulness via RAGAS.
- Recall@k em queries-canário por categoria.
- Latência p95.

### 7.3 Agente

Métricas qualitativas:
- Capacidade de selecionar a ferramenta correta dado o tipo de query (precisão de roteamento).
- Taxa de respostas com `Final Answer` válida (ausência de loop infinito).
- Latência média (impactada por `max_steps`).

---

## 8. Avaliação de segurança

**Documento mestre.** [docs/OWASP.md](docs/OWASP.md) — 5 ameaças mapeadas com referência direta ao código.
**Cenários de Red Team.** [docs/RED_TEAM_REPORT.md](docs/RED_TEAM_REPORT.md) — 5 cenários executáveis com payloads `curl` reais, alinhados às 5 ameaças.

### 8.1 Sumário do estado atual

| Categoria | Status | Severidade |
|-----------|--------|------------|
| Autenticação e autorização (API + observabilidade) | ❌ Ausente em todos os endpoints; Grafana com anonymous Viewer e senha default; Langfuse com `NEXTAUTH_SECRET`/`SALT` previsíveis | **Crítica** |
| CORS | ❌ `*` + `credentials=True` | **Alta** |
| Prompt injection (direta + indireta via `/ingest`, `/ingest_mlflow`, `fetch_news`) | ⚠️ Guardrail implementado mas **não integrado** em `app.py` | **Alta** |
| Data/Model poisoning | ❌ `/ingest` e `/ingest_mlflow` públicos; `overwrite=True` por default | **Crítica** |
| PII no output (e exfiltração para vLLM remoto) | ⚠️ Presidio implementado mas **não integrado**; prompt enviado integralmente à RunPod | **Alta** |
| Excessive agency do agente | ⚠️ Mitigação parcial (TOOL_MAP fechado, `max_steps`); sem auditoria persistente | **Alta** |
| Rate limiting / DoS / Custo | ❌ Ausente — cada chamada ao agente queima tokens da RunPod | Média |

### 8.2 Próximos passos
Roadmap em [docs/OWASP.md §4](docs/OWASP.md#4-próximos-passos-sugeridos), priorizado P0 → P2.

---

## 9. Avaliação ética, fairness e explicabilidade

**Documento mestre.** [docs/EXPLAINABILITY_FAIRNESS.md](docs/EXPLAINABILITY_FAIRNESS.md).

### 9.1 Explicabilidade — estado atual
| Camada | Hoje | Plano |
|--------|------|-------|
| Modelos preditivos | Apenas MAE/RMSE/MAPE | SHAP, Integrated Gradients (Captum), permutation importance |
| RAG | `_build_context` ([app.py:67-104](src/serving/app.py#L67-L104)) já estrutura `Fonte i:` com título e `fetched_at`, mas sem citação por sentença | Citações estruturadas `[1][2]`, faithfulness score (RAGAS) |
| Agente ReAct | Trace nativo do paradigma + fallbacks `rag_fast_path` / `rag_fallback` em [app.py:286-335](src/serving/app.py#L286-L335) | Sanitização do trace, marcador `generation_mode`, persistência no Langfuse |

### 9.2 Fairness — riscos identificados
| ID | Risco | Limiar / Métrica |
|----|-------|------------------|
| R1 | Performance gap por subgrupo de ticker (large vs small cap, setor) | Razão MAPE ≤ 1.5x |
| R2 | Drift de regime macroeconômico | PSI ≤ 0.2 |
| R3 | Cobertura desigual no RAG (BR vs internacional) | Recall@3 ≥ 0.7 |
| R4 | Resposta categórica rígida em `_generate_simulated_answer` | Documentar como decisão de produto ou substituir |
| R5 | Viés do LLM upstream (Qwen2.5 AWQ na RunPod, fallback `flan-t5-base`) | Bias eval em PT-BR |

### 9.3 Considerações éticas adicionais

- **Transparência ao usuário final.** Toda resposta com recomendação financeira deve carregar disclaimer obrigatório (ver §10).
- **Direito à revisão (LGPD Art. 20).** Decisões automatizadas devem permitir contestação por pessoa natural — endpoint `POST /lgpd/review` planejado.
- **Não discriminação.** Modelo não deve ser usado para precificar serviço ou negar acesso baseado em característica protegida — fora do escopo, mas importante balizar.

---

## 10. Limitações e riscos conhecidos

### 10.1 Capacidade técnica
- Modelos preditivos passaram a ser **multivariados** com features engenheiradas ([src/features/feature_engineering.py](src/features/feature_engineering.py)), mas ainda **não incorporam** macroeconomia, notícias ou dividendos como features explícitas.
- Janela temporal fixa configurada via [params.yaml](params.yaml) — não adapta a horizontes diferentes.
- LSTM treina apenas se `--keras` for passado ([train.py:384](src/models/train.py#L384)).
- LLM de fallback HF local (`google/flan-t5-base`) é **fraco em português** — saída pode ser incoerente; o caminho preferencial é o vLLM remoto na RunPod.
- Modo `simulated` retorna respostas **hardcoded** ([generator.py:569+](src/rag/generator.py#L569)) que **parecem** geradas mas são determinísticas por palavra-chave.
- Endpoint `/agent` aplica fallbacks ([app.py:286-335](src/serving/app.py#L286-L335)) que substituem a resposta do agente quando o formato ReAct falha — usuário pode não distinguir o caminho usado.

### 10.2 Operacional
- Índice FAISS é **global e em memória** — perdido a cada reinício.
- Sem persistência entre sessões (a ingestão inicial automática reconstrói notícias padrão a cada startup — [embedding.py:211-214](src/rag/embedding.py#L211-L214)).
- Sem isolamento entre usuários (ver [docs/RED_TEAM_REPORT.md §RT-03](docs/RED_TEAM_REPORT.md#rt-03--exfiltração-de-pii-incl-para-o-llm-remoto-na-runpod)).
- Variáveis globais de módulo no RAG ([embedding.py:47-52](src/rag/embedding.py#L47-L52)) impedem múltiplos workers paralelos sem cuidado adicional.
- Stack de observabilidade com **credenciais default** (Grafana `admin:datathon2024`, Langfuse `NEXTAUTH_SECRET` previsível, Postgres `langfuse:langfuse`) — não pode ser exposto na Internet sem hardening.

### 10.3 Disclaimer obrigatório

> Toda saída do sistema com sugestão de investimento deve ser acompanhada do seguinte aviso, conforme [docs/RED_TEAM_REPORT.md §RT-02](docs/RED_TEAM_REPORT.md#rt-02--data-poisoning-via-ingest-ingest_mlflow-e-fetch_news):
>
> *"Conteúdo educacional e experimental. Não constitui recomendação de investimento nos termos da Resolução CVM 39/2021. Consulte um analista de valores mobiliários autorizado antes de tomar decisões financeiras."*

---

## 11. Conformidade regulatória

| Norma | Status | Documento |
|-------|--------|-----------|
| **LGPD** (Lei 13.709/2018) | Plano de adequação elaborado, **execução pendente** | [docs/LGPD_PLAN.md](docs/LGPD_PLAN.md) |
| **Resolução CVM 39/2021** (analista de valores) | **Não aplicável** se uso restrito a propósito acadêmico/educacional + disclaimer | — |
| **Marco Civil da Internet** (Lei 12.965/2014) | Aplicável a logs e guarda — alinhado se logs sanitizados | — |
| **NIST AI RMF** | Referência adotada para estrutura de avaliação | — |
| **EU AI Act** | Não aplicável (sistema brasileiro), mas usado como referência para sistemas de alto risco | — |

---

## 12. Monitoramento e atualização

### 12.1 Métricas em produção (planejadas)

| Métrica | Fonte | Alerta |
|---------|-------|--------|
| Latência p95 por endpoint | Prometheus (`:9090`) + FastAPI middleware (`:8001`) | > 3s |
| Taxa de erro 5xx | Prometheus | > 1% |
| Tokens consumidos por hora (vLLM RunPod) | Langfuse + Prometheus | > baseline + 50% |
| Custo estimado por hora (vLLM RunPod) | Métrica derivada em Grafana | Limiar mensal configurado |
| PSI dos modelos preditivos | Job DVC diário | > 0.2 |
| Detecções do `InputGuardrail` | Logger estruturado / Langfuse | Pico anômalo |
| Detecções do `OutputGuardrail` (PII) | Logger estruturado | Qualquer detecção → alerta |
| Chamadas de tool por sessão (especialmente `fetch_news`) | Langfuse | > limite por tool |

### 12.2 Cadência de revisão

| Revisão | Frequência |
|---------|------------|
| System Card (este documento) | A cada release ou trimestral |
| Model Card | A cada novo treino do modelo |
| OWASP / Red Teaming | Trimestral + após mudança de superfície |
| LGPD Plan | Trimestral + após mudança regulatória |
| Fairness dashboard | Mensal |

### 12.3 Critérios para retirar de uso (sunset)

O sistema **deve** ser desativado se:
1. PSI dos modelos preditivos > 0.5 sustentado por 7 dias sem retreino.
2. Faithfulness RAG < 0.6 sustentado por 7 dias.
3. Incidente de segurança Crítico não remediado em 72h.
4. Vazamento de PII confirmado.

---

## 13. Histórico de versões

| Versão | Data | Mudança | Autor |
|--------|------|---------|-------|
| 1.0 | 2026-04-28 | Versão inicial do System Card | Grupo 05 |
| 1.1 | 2026-05-03 | Atualização para refletir: vLLM remoto na RunPod (`VLLM_BASE_URL`), endpoint `/ingest_mlflow`, stack de observabilidade Prometheus/Grafana/Langfuse, modelos preditivos multivariados, OWASP consolidado em 5 ameaças | Grupo 05 |

---

## Documentos relacionados

- [docs/MODEL_CARD.md](docs/MODEL_CARD.md) — Model Card detalhado dos modelos preditivos
- [docs/OWASP.md](docs/OWASP.md) — Mapeamento OWASP de ameaças
- [docs/RED_TEAM_REPORT.md](docs/RED_TEAM_REPORT.md) — Cenários de Red Teaming
- [docs/LGPD_PLAN.md](docs/LGPD_PLAN.md) — Plano de adequação à LGPD
- [docs/EXPLAINABILITY_FAIRNESS.md](docs/EXPLAINABILITY_FAIRNESS.md) — Explicabilidade e Fairness
- [README.md](README.md) — Documentação do projeto
