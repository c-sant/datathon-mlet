# System Card — Datathon MLET

**Versão:** 0.1.0
**Última atualização:** 2026-04-23
**Status:** Protótipo acadêmico (datathon FIAP — MLET)
**Repositório:** `datathon-mlet`

Este System Card descreve de forma estruturada as capacidades, limitações, riscos e controles do sistema. Segue o padrão de *system/model cards* adotado por Anthropic, OpenAI e Google, adaptado para o contexto de um sistema híbrido **RAG + previsão de séries temporais financeiras** com MLOps completo.

---

## 1. Sumário executivo

O sistema combina dois subsistemas integrados em um mesmo pipeline MLOps:

1. **Subsistema de previsão de ações** — treina e avalia modelos (PyTorch, Scikit-learn, Keras e Ensemble) sobre séries históricas de preços (`yfinance`) para projetar `Close` futuro. Rastreado em MLflow; versionado via DVC.
2. **Subsistema RAG (Retrieval-Augmented Generation)** — responde perguntas em português sobre mercado financeiro, recuperando contexto de notícias ingeridas e gerando resposta via LLM (vLLM/BentoML ou fallback local).

A interface externa é uma API **FastAPI** ([app/main.py](app/main.py)) e um serviço de geração exposto por BentoML. Guardrails de entrada/saída em [src/security/guardrails.py](src/security/guardrails.py) e [src/security/pii_detection.py](src/security/pii_detection.py).

### Ficha técnica

| Campo | Valor |
|-------|-------|
| Tipo de sistema | Pipeline MLOps (previsão) + RAG LLM (chat) |
| Domínio | Mercado financeiro brasileiro |
| Idioma principal | Português (pt-BR) |
| Modelo de geração (prod) | `Qwen/Qwen2.5-0.5B-Instruct-AWQ` (vLLM quantizado AWQ) |
| Modelo de geração (local CPU) | `facebook/opt-125m` (dev) / `facebook/opt-1.3b` (fallback) / modo `simulated` |
| Modelo de embeddings | `all-MiniLM-L6-v2` (SentenceTransformers, 384 dims) |
| Vector store | FAISS `IndexFlatL2` |
| Preditores | MLP PyTorch (64→32→1), MLP sklearn `(64, 32)`, Keras LSTM opcional, Ensemble |
| Chunking | 300 palavras, overlap 50 |
| Runtime | Python ≥3.10, <3.14 |
| Serving | FastAPI + BentoML + vLLM; Docker Compose |

---

## 2. Uso pretendido

### 2.1. Casos de uso alvo
- **Educacional/datathon**: demonstrar pipeline MLOps reprodutível (DVC + MLflow + CI/CD + Docker).
- **Exploração assistida**: responder perguntas contextuais sobre tendências de mercado recentes com base em notícias ingeridas.
- **Baseline de previsão**: comparar frameworks (PyTorch vs. Sklearn vs. Keras) com métricas padronizadas (MAE, RMSE, MAPE).

### 2.2. Fora de escopo (não usar para)
- **Aconselhamento financeiro personalizado ou recomendação de investimento a clientes**. O sistema não é um consultor/analista registrado na CVM.
- **Decisões automatizadas com efeito jurídico** sobre titulares sem revisão humana (Art. 20 LGPD).
- **Operações algorítmicas** (alta frequência, execução automática).
- **Análise em idiomas diferentes de português**.
- **Uso com dados pessoais sensíveis** (saúde, biometria, etc.).
- **Decisões de crédito, seguros ou contratação**.

### 2.3. Usuários previstos
Estudantes, pesquisadores e desenvolvedores explorando MLOps. Não há onboarding para usuários finais leigos — interface é API, não UI.

---

## 3. Arquitetura do sistema

```
                              ┌──────────────────────┐
Usuário ─── HTTP ──────────► │   FastAPI app/main   │
                              └──────────┬───────────┘
                                         │
                         ┌───────────────┼──────────────────┐
                         ▼               ▼                  ▼
                ┌────────────────┐ ┌────────────┐ ┌─────────────────┐
                │ InputGuardrail │ │  Retriever │ │ OutputGuardrail │
                │ (regex PI,     │ │  (FAISS    │ │ (Presidio PII)  │
                │  len ≤ 4096)   │ │   top-k)   │ └─────────────────┘
                └────────────────┘ └─────┬──────┘
                                         │
                              ┌──────────▼──────────┐
                              │  Generator.py       │
                              │  → BentoML → vLLM   │
                              │  → fallback local   │
                              │  → fallback simulado│
                              └──────────┬──────────┘
                                         │
                                         ▼
                                    Resposta pt-BR
```

Subsistema de previsão (offline):
```
yfinance ──► data/ingest.py ──► DVC ──► src/models/baseline.py ──► MLflow
                                                │
                                                ▼
                                 evaluation/benchmark.py (CSV + gráficos)
                                                │
                                                ▼
                                 ci-cd/register_model.py (MLflow Registry)
```

Arquivos principais:
- API: [app/main.py](app/main.py), [src/serving/app.py](src/serving/app.py)
- RAG: [src/rag/embedding.py](src/rag/embedding.py), [src/rag/retriever.py](src/rag/retriever.py), [src/rag/generator.py](src/rag/generator.py)
- Agente: [src/agent/react_agent.py](src/agent/react_agent.py), [src/agent/rag_pipeline.py](src/agent/rag_pipeline.py)
- Segurança: [src/security/guardrails.py](src/security/guardrails.py), [src/security/pii_detection.py](src/security/pii_detection.py)
- Previsão: [src/models/baseline.py](src/models/baseline.py), [src/models/train.py](src/models/train.py)
- Avaliação: [evaluation/benchmark.py](evaluation/benchmark.py)
- Orquestração: `dvc.yaml`, `params.yaml`, [.github/workflows/ci-cd.yml](.github/workflows/ci-cd.yml), [.github/workflows/retrain.yml](.github/workflows/retrain.yml)
- Deploy: [docker-compose.yaml](docker-compose.yaml), [docker/docker-compose.bento.remote.yml](docker/docker-compose.bento.remote.yml)

---

## 4. Componentes de modelo

### 4.1. LLM de geração (RAG)

| Atributo | Produção (GPU remota) | Local CPU | Simulado |
|----------|----------------------|-----------|----------|
| Modelo | `Qwen/Qwen2.5-0.5B-Instruct-AWQ` | `facebook/opt-125m` / `opt-1.3b` | Regras em [src/rag/generator.py:118-152](src/rag/generator.py#L118-L152) |
| Serving | vLLM + BentoML (RunPod) | Transformers pipeline | Nenhum |
| Quantização | AWQ 4-bit | Nenhuma (FP32) | N/A |
| Contexto | até 32k (teórico) | 2048 | N/A |
| Licença | Apache 2.0 (Qwen) / MIT (OPT) | idem | N/A |
| Prompt template | `"Pergunta: {query}\n\nContexto: {context}\n\nResponda em português..."` | idem | Heurística por palavra-chave |

**Decoding padrão:** `temperature=0.7`, `do_sample=True`, `max_new_tokens=256` ([src/rag/generator.py:106](src/rag/generator.py#L106)).

### 4.2. Embedder

- Modelo: `all-MiniLM-L6-v2` (SentenceTransformers).
- Dimensão: 384.
- Treinamento original: corpus multilíngue de similaridade semântica.
- Adequação a pt-BR: moderada — não é especializado em português; embeddings menos densos em vocabulário financeiro brasileiro que num modelo dedicado.

### 4.3. Preditores de séries temporais

Todos treinados sobre `Close` normalizado com `MinMaxScaler(0, 1)` e janela configurável (padrão 90 dias):

| Framework | Arquitetura | Hiperparâmetros |
|-----------|-------------|-----------------|
| PyTorch | MLP 3 camadas (`input→64→32→1`, ReLU) | Adam lr=1e-3, 50 epochs, MSE loss |
| Sklearn | `MLPRegressor(64, 32)` | `max_iter=500` |
| Keras | LSTM (opcional, via `--keras`) | carregado de artefato `.keras` |
| Baseline | Último valor da janela (naive) | sem parâmetros |
| Ensemble | Média aritmética dos acima | — |

Métricas reportadas: **MAE, RMSE, MAPE** (em escala original, via `scaler.inverse_transform`) — ver [src/models/baseline.py:86-100](src/models/baseline.py#L86-L100).

### 4.4. Agente ReAct (experimental)

[src/agent/react_agent.py](src/agent/react_agent.py) expõe um loop de raciocínio com ferramentas (search, retrieve). **Não ativado por padrão no endpoint principal** — uso explícito via `rag_pipeline.py agent`. Ver §7 para limitações.

---

## 5. Dados

### 5.1. Corpus RAG
- **Fonte**: notícias financeiras brasileiras coletadas por `newspaper3k`, configuradas em [src/rag/data_loader.py](src/rag/data_loader.py).
  - `seudinheiro.com/mercados`
  - `einvestidor.estadao.com.br/mercado`
  - `infomoney.com.br/mercados`
- **Volume**: variável (dependente da coleta no momento da execução — tipicamente dezenas de artigos).
- **Licenciamento**: conteúdo jornalístico de terceiros. **Uso restrito a pesquisa acadêmica/datathon** (fair use educacional). Distribuição pública do índice FAISS populado **não é autorizada**.
- **Atualização**: sob demanda, via `POST /ingest` ou execução de [src/rag/embedding.py](src/rag/embedding.py).
- **Viés esperado**: perspectiva predominantemente brasileira, foco em renda variável/mercado local, possível tendência editorial dos veículos.

### 5.2. Séries financeiras
- **Fonte**: Yahoo Finance via `yfinance>=0.2.40` (declarado em [pyproject.toml](pyproject.toml#L23)).
- **Campos usados**: `Close` diário.
- **Tickers**: parametrizáveis (`--ticker`); padrão sujeito a `params.yaml`.
- **Risco de qualidade**: dados da API podem ter ajustes históricos retroativos (splits, dividendos) que alteram resultados entre execuções.

### 5.3. PII / dados pessoais
O corpus **pode conter nomes de analistas, executivos, e eventualmente PII em comentários**. Controles de mitigação:
- Detecção: [src/security/pii_detection.py](src/security/pii_detection.py) cobre CPF, CNPJ, e-mail, telefone BR, cartão, CEP, IP.
- Sanitização saída: `OutputGuardrail` via Presidio ([src/security/guardrails.py:56-92](src/security/guardrails.py#L56-L92)).
- **Gap atual**: sanitização não é aplicada na ingestão. Ver [docs/LGPD.md](docs/LGPD.md).

---

## 6. Avaliação

### 6.1. Previsão (subsistema quantitativo)

Pipeline em [evaluation/benchmark.py](evaluation/benchmark.py):
- Consulta runs MLflow do experimento `previsao_acoes`.
- Produz `reports/metrics_comparison.csv`, `reports/metrics.json` e gráficos.
- Métricas por framework: `mae_{framework}`, `rmse_{framework}`, `mape_{framework}`.
- **Benchmark atual**: ≥3 configurações documentadas (PyTorch, Sklearn, Keras) conforme [README.md](README.md).

### 6.2. RAG (subsistema qualitativo)

- **Métricas quantitativas**: não implementadas (sem ground truth curado). O projeto reporta métricas operacionais (latência, disponibilidade) mas não **faithfulness**, **context relevance** ou **answer correctness**.
- **Teste ofensivo**: [tests/test_guardrails.py](tests/test_guardrails.py) valida bloqueio de prompt injection e remoção de PII.
- **Teste de integração**: [tests/test_api.py](tests/test_api.py), [tests/test_ingest.py](tests/test_ingest.py), [tests/test_plot_metrics.py](tests/test_plot_metrics.py).
- **Cobertura**: medida via `pytest-cov` (configuração em [pyproject.toml](pyproject.toml)).

**Lacuna**: não há benchmark automatizado de qualidade de resposta (ex.: RAGAS, TruLens). É a próxima evolução recomendada antes de qualquer uso externo.

### 6.3. Red-team / safety
- Testes manuais de prompt injection registrados em [tests/test_guardrails.py](tests/test_guardrails.py) (padrões `ignore previous instructions`, `you are now`, etc.).
- Não há avaliação sistemática de: jailbreaks multilíngues, injeção indireta via corpus ingerido, ataques de embedding inversion, tool misuse no agente ReAct.

---

## 7. Limitações conhecidas

### 7.1. Capacidade do LLM
- `Qwen2.5-0.5B` (500M parâmetros) é **pequeno por padrão de 2026**. Capacidade limitada em raciocínio multi-hop, matemática e seguir instruções complexas.
- `facebook/opt-125m` (fallback CPU) é ainda mais fraco — usado apenas para desenvolvimento.
- O **modo `simulated`** em [src/rag/generator.py:118-152](src/rag/generator.py#L118-L152) retorna respostas **heurísticas baseadas em palavras-chave**, não em geração real. Útil para demo, mas **não representa a qualidade do pipeline em produção** — é importante não interpretar suas respostas como saída do LLM.

### 7.2. RAG
- Chunking por **contagem de palavras** (300/50 overlap) — pode quebrar sentenças no meio e degradar retrieval.
- `IndexFlatL2` é **busca exata O(n)** — ok para dezenas de milhares de chunks, não escala além.
- **Sem reranker** — top-k é determinado só pela distância L2 do MiniLM, sensível a atalhos lexicais.
- **Sem citação de fontes automática** na resposta final — o contexto é usado, mas o LLM não é obrigado a referenciar `doc_id`.
- Embedder não é otimizado para português financeiro → **recall subótimo** em jargão brasileiro.

### 7.3. Previsão
- Modelos usam **só o histórico de `Close`** — ignoram volume, indicadores técnicos, dados fundamentais, macro.
- Janela e horizonte são parametrizados, mas **não validados com walk-forward**. Risco de *lookahead bias* se o pipeline for adaptado sem cuidado.
- Ensemble é **média simples**, sem ponderação por qualidade individual.
- **Benchmark baseline (naive) é competitivo**: MLPs rasos dificilmente superam "prever o último valor" em mercados eficientes.

### 7.4. Agente ReAct
- **Sem sandboxing**: tools têm acesso potencialmente irrestrito a I/O e rede.
- **Sem limite formal de iterações** — risco de loop custoso controlado por prompt adversarial.
- Não integrado aos guardrails por padrão.

### 7.5. Guardrails
- Lista de regex de prompt injection em [src/security/guardrails.py:15-28](src/security/guardrails.py#L15-L28) é **fácil de contornar** com Unicode/obfuscação/tradução.
- Presidio com `language="pt"` não cobre 100% das entidades BR (falta `BR_CNPJ`, `CREDIT_CARD`, `IBAN_CODE` na lista atual).
- Sanitização **só na saída**: não protege contra vazamento via canais laterais (logs, MLflow artifacts, etc.).

---

## 8. Riscos e mitigações

Ver documentos dedicados:
- **Técnicos (OWASP LLM Top 10 + API)**: [docs/OWASP.md](docs/OWASP.md).
- **Jurídicos (LGPD)**: [docs/LGPD.md](docs/LGPD.md).

Resumo dos principais riscos residuais:

| Risco | Severidade | Mitigação ativa | Lacuna |
|-------|------------|-----------------|--------|
| Prompt injection direto | Alta | Regex + limite 4096 chars | Contorno por Unicode; classificador semântico ausente |
| Prompt injection indireto via corpus | Alta | — | Nenhuma — corpus ingerido sem filtro |
| Vazamento de PII | Alta | Presidio na saída | Não aplicado na ingestão; logs guardam prompt cru |
| Transferência internacional (Art. 33 LGPD) | Alta | Caminho CPU local disponível | Sem sanitização pré-envio ao vLLM RunPod |
| Alucinação financeira com aparência de conselho | Alta | Aviso em [README.md](README.md) | Sem disclaimer automático na resposta |
| Denial of wallet (GPU RunPod) | Média | — | Sem rate limit; sem budget cap |
| Poisoning do corpus | Média | DVC + MLflow auditam mudanças | Sem assinatura; sem allow-list de fontes |
| Discriminação em recomendações | Média | — | Sem teste de fairness; ver [docs/LGPD.md](docs/LGPD.md) §4.4 |

---

## 9. Considerações éticas

- **Decisão automatizada**: saída do RAG pode influenciar decisões financeiras. Recomenda-se disclaimer visível (`"conteúdo educacional, não é recomendação de investimento"`) e revisão humana obrigatória.
- **Viés de fonte**: três veículos brasileiros específicos moldam a visão do sistema. Perspectivas de veículos internacionais, acadêmicos ou críticos estão sub-representadas.
- **Acesso desigual**: sistema é em português. Usuários não fluentes ou com deficiências sensoriais não estão previstos no design atual.
- **Transparência**: respostas do modo `simulated` podem ser confundidas com saída de LLM real. Deve haver indicação clara do modo ativo.

---

## 10. Operação e monitoramento

### 10.1. Observabilidade
- **MLflow** (`sqlite:///mlflow/mlflow.db`): experimentos, métricas, artefatos.
- **Logs Python** (`logging` padrão): WARNING em eventos de guardrail; INFO em auditoria PII.
- **Health check**: `GET /health` em [app/main.py:23-29](app/main.py#L23-L29) retorna status + versão do modelo.
- **Gap**: sem métricas Prometheus/OpenTelemetry; sem alerting.

### 10.2. Atualização e retrain
- **CI/CD**: [.github/workflows/ci-cd.yml](.github/workflows/ci-cd.yml) roda testes + lint em PR.
- **Retrain**: [.github/workflows/retrain.yml](.github/workflows/retrain.yml) executa pipeline DVC.
- **Registro de modelo**: [ci-cd/register_model.py](ci-cd/register_model.py) publica no MLflow Model Registry.

### 10.3. Canais de suporte
- Issues no GitHub (definir repositório canônico).
- Canal LGPD/DPO: a nomear (ver [docs/LGPD.md](docs/LGPD.md) §1).
- Report de vulnerabilidade: abrir issue marcando como `security` — **evitar PoCs públicos antes de fix**.

---

## 11. Dependências e supply chain

Dependências declaradas em [pyproject.toml](pyproject.toml):

| Categoria | Pacotes |
|-----------|---------|
| Core | `numpy>=1.26`, `pandas>=2.2`, `matplotlib>=3.8` |
| ML | `scikit-learn>=1.4`, `joblib>=1.4` |
| Dados | `yfinance>=0.2.40` |
| DL | `torch>=2.2`, `tensorflow>=2.15` |
| MLOps | `mlflow>=2.12`, `dvc>=3.50` |
| Segurança | `presidio-analyzer>=2.2.362`, `presidio-anonymizer>=2.2.362` |
| Dev/test | `black`, `ruff`, `pytest`, `pytest-cov`, `httpx` |

Serviços externos invocados em runtime:
- **Hugging Face Hub** — download de `all-MiniLM-L6-v2`, `Qwen2.5-0.5B-Instruct-AWQ`, `facebook/opt-*`.
- **RunPod** — GPU para vLLM (transferência internacional potencial).
- **Yahoo Finance** (via `yfinance`) — séries históricas.
- **Veículos de notícias** (via `newspaper3k`) — corpus RAG.

**Pre-commit**: [.pre-commit-config.yaml](.pre-commit-config.yaml) configurado. **Gap**: sem `pip-audit` / `safety` no CI.

---

## 12. Privacidade e conformidade

- **LGPD (Brasil)**: plano detalhado em [docs/LGPD.md](docs/LGPD.md). Status atual: **não conforme para uso produtivo** (gaps P0 em logs, transferência internacional, direitos do titular).
- **GDPR (UE)**: não analisado. Se houver titulares UE, exige revisão separada.
- **Dados financeiros**: sem tratamento de dados pessoais sensíveis no projeto-base. Se adicionar perfil do investidor, precisa de consentimento (Art. 7º I + regulação CVM).

---

## 13. Versionamento deste System Card

| Versão | Data | Mudanças |
|--------|------|----------|
| 0.1.0 | 2026-04-23 | Versão inicial cobrindo RAG + previsão + guardrails + LGPD/OWASP links |

**Política de atualização**: este documento deve ser revisado:
- Em toda mudança de modelo (LLM, embedder, preditor).
- Em toda mudança de fonte de dados.
- Em todo incidente de segurança ou privacidade.
- A cada release minor (0.x.0).

---

## 14. Apêndice A — Matriz de conformidade com frameworks

| Framework | Seção correspondente |
|-----------|---------------------|
| NIST AI RMF — Govern | §1, §10, §13 |
| NIST AI RMF — Map | §2, §3, §5 |
| NIST AI RMF — Measure | §6, §7 |
| NIST AI RMF — Manage | §8, §10.2, [docs/OWASP.md](docs/OWASP.md) |
| OWASP Top 10 LLM 2025 | [docs/OWASP.md](docs/OWASP.md) |
| LGPD (Lei 13.709/2018) | [docs/LGPD.md](docs/LGPD.md) |
| Microsoft MLOps Maturity — Experiment Mgmt | Stage 3 (ver [README.md](README.md)) |

## 15. Apêndice B — Glossário mínimo

- **RAG**: Retrieval-Augmented Generation — geração baseada em contexto recuperado externamente.
- **AWQ**: Activation-aware Weight Quantization — quantização pós-treino que preserva outliers.
- **PII**: Personally Identifiable Information.
- **ROPA**: Record of Processing Activities (Art. 37 LGPD).
- **RIPD/DPIA**: Relatório de Impacto à Proteção de Dados (Art. 38 LGPD).
- **MLOps**: Machine Learning Operations — disciplina que trata de CI/CD, versionamento e observabilidade de modelos.
