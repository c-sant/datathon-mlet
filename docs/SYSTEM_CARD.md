# System Card — Datathon MLET (Grupo 05)

Documento que descreve o **sistema completo** colocado em operação: o que faz, como funciona, em que contextos pode ser usado com responsabilidade e quais riscos foram avaliados. Inspirado em System Cards publicados por OpenAI e Anthropic e nas diretrizes do **NIST AI Risk Management Framework**.

> **Status do documento.** v1.0 — 2026-04-28
> **Versão do sistema descrita.** Branch `docs` em commit `5553fb3` (snapshot atual do repositório).
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
| Demonstração acadêmica de pipeline MLOps (treino + tracking + deploy) | Avaliadores FIAP, banca | [src/models/train.py](src/models/train.py), MLflow, [dvc.yaml](dvc.yaml) |
| Comparação de frameworks (PyTorch vs Sklearn vs Keras) sobre série univariada | Estudantes, professores | [benchmark.py](benchmark.py) |
| Exploração de RAG com FAISS sobre corpus financeiro pequeno | Estudantes | [src/rag/](src/rag/) |
| Exploração de agente ReAct com chamada de ferramentas | Estudantes | [src/agent/react_agent.py](src/agent/react_agent.py) |

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
┌───────────────────────────────────────────────────────────────┐
│                       Cliente HTTP                            │
│                  (api_test.html / curl)                       │
└────────────┬──────────────────────────┬───────────────────────┘
             │                          │
             ▼                          ▼
   ┌────────────────────┐    ┌────────────────────┐
   │   /ingest          │    │   /query  /agent   │
   │   (RAG ingestion)  │    │   (RAG + Agent)    │
   └────────┬───────────┘    └─────┬──────────────┘
            │                      │
            ▼                      ▼
   ┌────────────────────┐    ┌────────────────────┐
   │ embedder           │    │ Agente ReAct       │
   │ (SentenceTransfor- │    │ (react_agent.py)   │
   │  mers all-MiniLM)  │    │ ┌────────────────┐ │
   └────────┬───────────┘    │ │ search_docs    │ │
            │                │ │ fetch_news     │ │
            ▼                │ │ summarize      │ │
   ┌────────────────────┐    │ └────────────────┘ │
   │ FAISS IndexFlatL2  │◄───┤                    │
   │ (in-memory, global)│    └─────┬──────────────┘
   └────────────────────┘          │
                                   ▼
                         ┌──────────────────────┐
                         │ Generator            │
                         │ (BentoML/vLLM remoto │
                         │  ou HF local         │
                         │  ou simulated)       │
                         └──────────────────────┘

   ┌──────────────────────────────────────────┐
   │ Modelos preditivos (offline / batch)     │
   │ MLP PyTorch · MLP Sklearn · LSTM Keras   │
   │ Treinados via train.py, tracking MLflow  │
   └──────────────────────────────────────────┘
```

Detalhamento: [src/serving/app.py](src/serving/app.py), [src/agent/react_agent.py](src/agent/react_agent.py), [src/rag/](src/rag/).

---

## 5. Componentes

### 5.1 API HTTP (FastAPI)
| Atributo | Valor |
|----------|-------|
| Arquivo | [src/serving/app.py](src/serving/app.py) |
| Endpoints | `POST /ingest`, `GET /query`, `POST /agent` |
| Autenticação | **Nenhuma** (gap conhecido — ver [docs/OWASP.md §3.9](docs/OWASP.md#39-api12023--a012021--broken-access-control)) |
| CORS | `allow_origins=["*"]` (gap — ver [docs/OWASP.md §3.10](docs/OWASP.md#310-a052021--security-misconfiguration-cors-permissivo)) |

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
| Embedder | `SentenceTransformer("all-MiniLM-L6-v2")` | [src/rag/embedding.py:31-44](src/rag/embedding.py#L31-L44) |
| Chunker | Janela 300 palavras, overlap 50 | [src/rag/embedding.py:12-21](src/rag/embedding.py#L12-L21) |
| Vector Store | `faiss.IndexFlatL2` (in-memory, global) | [src/rag/embedding.py:75-76](src/rag/embedding.py#L75-L76) |
| Retriever | Top-K por distância L2 | [src/rag/retriever.py](src/rag/retriever.py) |
| Generator | Bento/vLLM (`Qwen2.5-0.5B-Instruct-AWQ`) ou HF (`opt-1.3b`/`distilgpt2`) ou `simulated` | [src/rag/generator.py](src/rag/generator.py) |

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
| Tracking | MLflow local (`sqlite:///mlflow.db`) |
| Versionamento de dados | DVC ([dvc.yaml](dvc.yaml)) |
| Containerização | Docker Compose (`docker/docker-compose.yml`) |
| Inferência LLM remota | RunPod com vLLM AWQ (validada — [README.md:119-128](README.md#L119-L128)) |
| Gerenciamento de deps | Poetry ([pyproject.toml](pyproject.toml)) |

---

## 6. Dados

### 6.1 Dados de treino dos modelos preditivos
- **Fonte.** CSV com cotações históricas (coluna `Close`) — caminho configurável via `--data-path` ([train.py:315](src/models/train.py#L315)).
- **Janela.** Configurável (`--janela`), default em [config/](config/).
- **Split.** Temporal 80/20 ([train.py:169-171](src/models/train.py#L169-L171)).
- **Pré-processamento.** `MinMaxScaler` no fechamento ([train.py:117-118](src/models/train.py#L117-L118)).

### 6.2 Dados ingeridos no RAG
- **Fonte primária.** Notícias carregadas via [src/rag/data_loader.py](src/rag/data_loader.py) (sem whitelist explícita — gap em [docs/OWASP.md §3.3](docs/OWASP.md#33-llm042025--data-and-model-poisoning)).
- **Fonte secundária.** Documentos enviados pelo cliente em `POST /ingest`.
- **Persistência.** Apenas em memória do processo. **Não há** banco persistente.

### 6.3 Dados pessoais
Inventário completo em [docs/LGPD_PLAN.md §2](docs/LGPD_PLAN.md#2-mapeamento-de-dados-pessoais-registro-de-operações-de-tratamento).

---

## 7. Avaliação de capacidades

### 7.1 Modelos preditivos

Métricas de teste reportadas pelo último benchmark (ver MLflow / [benchmark.py](benchmark.py)):

| Modelo | MAE ↓ | RMSE ↓ | MAPE ↓ |
|--------|-------|--------|--------|
| Baseline (persistence) | tracking via `mae_baseline` | `rmse_baseline` | `mape_baseline` |
| MLP PyTorch | `mae_pytorch` | `rmse_pytorch` | `mape_pytorch` |
| MLP Sklearn | `mae_sklearn` | `rmse_sklearn` | `mape_sklearn` |
| LSTM Keras | `mae_keras` | `rmse_keras` | `mape_keras` |

> Os valores numéricos são lidos do MLflow do último run. O System Card **não** os congela porque variam por ticker e janela; consulte o run mais recente ou o [Model Card](docs/MODEL_CARD.md).

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

**Documento mestre.** [docs/OWASP.md](docs/OWASP.md) — 10 ameaças mapeadas com referência direta ao código.
**Cenários de Red Team.** [docs/RED_TEAMING.md](docs/RED_TEAMING.md) — 5 cenários executáveis com payloads `curl` reais.

### 8.1 Sumário do estado atual

| Categoria | Status | Severidade |
|-----------|--------|------------|
| Autenticação e autorização | ❌ Ausente em todos os endpoints | **Crítica** |
| CORS | ❌ `*` + `credentials=True` | **Alta** |
| Prompt injection (direta) | ⚠️ Guardrail implementado mas **não integrado** | **Alta** |
| Prompt injection (indireta via `/ingest`) | ❌ Sem mitigação | **Crítica** |
| PII no output | ⚠️ Presidio implementado mas **não integrado** | **Alta** |
| Rate limiting / DoS | ❌ Ausente | Média |
| Excessive agency do agente | ⚠️ Mitigação parcial (TOOL_MAP fechado) | Média |
| Misinformation financeira | ❌ Sem disclaimer obrigatório | **Crítica** |

### 8.2 Próximos passos
Roadmap em [docs/OWASP.md §4](docs/OWASP.md#4-próximos-passos-sugeridos), priorizado P0 → P2.

---

## 9. Avaliação ética, fairness e explicabilidade

**Documento mestre.** [docs/EXPLAINABILITY_FAIRNESS.md](docs/EXPLAINABILITY_FAIRNESS.md).

### 9.1 Explicabilidade — estado atual
| Camada | Hoje | Plano |
|--------|------|-------|
| Modelos preditivos | Apenas MAE/RMSE/MAPE | SHAP, Integrated Gradients (Captum), permutation importance |
| RAG | `context` concatenado bruto, sem citação | Citações estruturadas `[1][2]`, faithfulness score |
| Agente ReAct | Trace nativo do paradigma | Sanitização do trace, marcador `generation_mode: simulated` |

### 9.2 Fairness — riscos identificados
| ID | Risco | Limiar / Métrica |
|----|-------|------------------|
| R1 | Performance gap por subgrupo de ticker (large vs small cap, setor) | Razão MAPE ≤ 1.5x |
| R2 | Drift de regime macroeconômico | PSI ≤ 0.2 |
| R3 | Cobertura desigual no RAG (BR vs internacional) | Recall@3 ≥ 0.7 |
| R4 | Resposta categórica rígida em `_generate_simulated_answer` | Documentar como decisão de produto ou substituir |
| R5 | Viés do LLM upstream (Qwen, OPT) | Bias eval em PT-BR |

### 9.3 Considerações éticas adicionais

- **Transparência ao usuário final.** Toda resposta com recomendação financeira deve carregar disclaimer obrigatório (ver §10).
- **Direito à revisão (LGPD Art. 20).** Decisões automatizadas devem permitir contestação por pessoa natural — endpoint `POST /lgpd/review` planejado.
- **Não discriminação.** Modelo não deve ser usado para precificar serviço ou negar acesso baseado em característica protegida — fora do escopo, mas importante balizar.

---

## 10. Limitações e riscos conhecidos

### 10.1 Capacidade técnica
- Modelos preditivos são **univariados** — só usam histórico de `Close`. Ignoram volume, macroeconomia, notícias, dividendos, splits.
- Janela temporal fixa configurada — não adapta a horizontes diferentes.
- LSTM treina apenas se `--keras` for passado ([train.py:252](src/models/train.py#L252)).
- LLM padrão de fallback (`facebook/opt-1.3b`, `distilgpt2`) é **fraco em português** — frequente saída incoerente.
- Modo `simulated` retorna respostas **hardcoded** ([generator.py:137-171](src/rag/generator.py#L137-L171)) que **parecem** geradas mas são determinísticas por palavra-chave.

### 10.2 Operacional
- Índice FAISS é **global e em memória** — perdido a cada reinício.
- Sem persistência entre sessões.
- Sem isolamento entre usuários (ver [docs/RED_TEAMING.md §RT-03](docs/RED_TEAMING.md#rt-03--exfiltração-de-pii-cross-tenant-pelo-índice-faiss-global)).
- Variáveis globais de módulo no RAG ([embedding.py:46-49](src/rag/embedding.py#L46-L49)) impedem múltiplos workers paralelos sem cuidado adicional.

### 10.3 Disclaimer obrigatório

> Toda saída do sistema com sugestão de investimento deve ser acompanhada do seguinte aviso, conforme [docs/RED_TEAMING.md §RT-04](docs/RED_TEAMING.md#rt-04--manipulação-de-recomendação-financeira-pump--dump-assistido-por-ia):
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
| Latência p95 por endpoint | Prometheus + FastAPI middleware | > 3s |
| Taxa de erro 5xx | Prometheus | > 1% |
| Tokens consumidos por hora (vLLM) | Logs vLLM | > baseline + 50% |
| PSI dos modelos preditivos | Job DVC diário | > 0.2 |
| Detecções do `InputGuardrail` | Logger estruturado | Pico anômalo |
| Detecções do `OutputGuardrail` (PII) | Logger estruturado | Qualquer detecção → alerta |

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

---

## Documentos relacionados

- [docs/MODEL_CARD.md](docs/MODEL_CARD.md) — Model Card detalhado dos modelos preditivos
- [docs/OWASP.md](docs/OWASP.md) — Mapeamento OWASP de ameaças
- [docs/RED_TEAMING.md](docs/RED_TEAMING.md) — Cenários de Red Teaming
- [docs/LGPD_PLAN.md](docs/LGPD_PLAN.md) — Plano de adequação à LGPD
- [docs/EXPLAINABILITY_FAIRNESS.md](docs/EXPLAINABILITY_FAIRNESS.md) — Explicabilidade e Fairness
- [README.md](README.md) — Documentação do projeto
