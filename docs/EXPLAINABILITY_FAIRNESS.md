# Explicabilidade e Fairness — Datathon MLET (Grupo 05)

Documento que **operacionaliza** os requisitos de explicabilidade (XAI) e *fairness* sobre o sistema atual, composto por três camadas com perfis de risco distintos:

1. **Modelos preditivos de regressão** sobre série temporal de preços (PyTorch MLP, Sklearn MLP, Keras LSTM) — ver [src/models/train.py](src/models/train.py).
2. **RAG** com índice FAISS sobre documentos financeiros — ver [src/rag/](src/rag/).
3. **Agente ReAct** que orquestra ferramentas e gera recomendações — ver [src/agent/react_agent.py](src/agent/react_agent.py).

Todas as camadas produzem ou influenciam **recomendações financeiras**, o que ativa o **Art. 20 da LGPD** (direito à revisão de decisões automatizadas) e o **Art. 6º, IX** (princípio da não discriminação). Este documento é complementar a [docs/OWASP.md](docs/OWASP.md), [docs/RED_TEAM_REPORT.md](docs/RED_TEAM_REPORT.md) e [docs/LGPD_PLAN.md](docs/LGPD_PLAN.md).

---

## Sumário

1. [Por que importa neste projeto](#1-por-que-importa-neste-projeto)
2. [Explicabilidade — Camada 1: modelos preditivos](#2-explicabilidade--camada-1-modelos-preditivos)
3. [Explicabilidade — Camada 2: RAG (recuperação)](#3-explicabilidade--camada-2-rag-recuperação)
4. [Explicabilidade — Camada 3: agente ReAct e geração](#4-explicabilidade--camada-3-agente-react-e-geração)
5. [Fairness — riscos identificados e métricas](#5-fairness--riscos-identificados-e-métricas)
6. [Plano de implementação](#6-plano-de-implementação)
7. [Métricas e dashboard](#7-métricas-e-dashboard)
8. [Governança e ciclo de revisão](#8-governança-e-ciclo-de-revisão)
9. [Referências](#9-referências)

---

## 1. Por que importa neste projeto

| Dimensão | Risco específico do projeto | Onde no código |
|----------|-----------------------------|----------------|
| Decisão automatizada com efeito financeiro | Recomendações de ações/alocação | [src/rag/generator.py:569+](src/rag/generator.py#L569), [src/agent/react_agent.py:60-78](src/agent/react_agent.py#L60-L78) |
| Modelo de séries temporais sem feature importance | MLP/LSTM tratam preço como única entrada — usuário não sabe **por que** previu X | [src/models/train.py:38-49](src/models/train.py#L38-L49), [src/models/train.py:262+](src/models/train.py#L262) |
| Recuperação RAG sem citação visível na resposta | `/query` retorna `answer` desacoplado das fontes | [src/serving/app.py:165-184](src/serving/app.py#L165-L184), [src/serving/app.py:244-262](src/serving/app.py#L244-L262) |
| Agente ReAct com trace exposto, mas sem garantia de fidelidade | Trace pode ser *post-hoc*, não causal | [src/agent/react_agent.py:104-170](src/agent/react_agent.py#L104-L170) |
| Viés potencial em recomendações | Concentração em ativos *blue-chip* / mercado brasileiro / horário comercial dos dados de treino | [src/rag/generator.py:569+](src/rag/generator.py#L569) |
| Decisões automatizadas (LGPD Art. 20) | Direito à revisão por pessoa natural | — não implementado |

---

## 2. Explicabilidade — Camada 1: modelos preditivos

### 2.1 Estado atual

Três modelos treinados sobre janela multivariada de features tratadas ([src/models/train.py:160](src/models/train.py#L160), [src/features/feature_engineering.py](src/features/feature_engineering.py)):

| Modelo | Arquitetura | Saída interpretável hoje? |
|--------|-------------|---------------------------|
| PyTorch MLP | 3 camadas densas (input → 64 → 32 → 1) — [train.py:38-49](src/models/train.py#L38-L49) | Não |
| Sklearn MLPRegressor | `hidden_layer_sizes=(64, 32)` — [train.py:364](src/models/train.py#L364) | Não |
| Keras LSTM | LSTM(50, return_sequences=True) → LSTM(50) → Dense(1) — [train.py:390-392](src/models/train.py#L390-L392) | Não |
| Baseline | `y_pred = X_test[:, -1]` (último valor) — [src/models/baseline.py](src/models/baseline.py) | Sim (trivial) |

**Métricas atuais.** MAE, RMSE, MAPE registradas no MLflow via pipeline DVC. Split temporal 80/20 logado em [train.py:317](src/models/train.py#L317). Métricas de **erro**, mas **não de explicabilidade**.

### 2.2 Técnicas a implementar

| Técnica | Aplica a | Saída | Esforço |
|---------|----------|-------|---------|
| **Feature importance via permutation** | Sklearn MLP | Quão sensível o modelo é a cada lag (`t-1`, `t-2`, ..., `t-N`) | Baixo (sklearn nativo) |
| **SHAP (KernelExplainer / DeepExplainer)** | PyTorch MLP, Keras LSTM | Contribuição de cada lag para a previsão de cada ponto | Médio |
| **Integrated Gradients** | PyTorch MLP | Atribuição diferenciável (mais consistente que SHAP em redes profundas) | Médio (Captum) |
| **Attention weights** | Adicionar atenção ao LSTM (opcional) | Visualização nativa de quais dias foram considerados | Alto (refator) |
| **Counterfactual explanations** | Todos | "O preço previsto subiria 2% se o lag t-3 fosse 5% maior" | Médio |
| **Confidence intervals via MC Dropout / Bootstrap** | Keras LSTM (já tem Dropout), MLP | Banda de incerteza ao redor da previsão pontual | Médio |

### 2.3 Output esperado por previsão

Cada chamada de inferência deve devolver, além do número:

```json
{
  "ticker": "PETR4.SA",
  "prediction": 38.42,
  "confidence_interval_95": [36.10, 40.74],
  "top_features": [
    {"lag": "t-1", "shap_value": 0.71, "direction": "positive"},
    {"lag": "t-2", "shap_value": 0.18, "direction": "positive"},
    {"lag": "t-7", "shap_value": -0.09, "direction": "negative"}
  ],
  "model_version": "pytorch_v3",
  "training_window": "2018-01-01 → 2024-12-31",
  "data_drift_alert": false
}
```

### 2.4 Limites a comunicar ao usuário (model card)

- O modelo prevê o **próximo fechamento**, não retorno acumulado nem volatilidade.
- Treinado em **uma única série univariada** — não considera macroeconomia, notícias, dividendos.
- **MAPE de N%** sobre teste — diferente do erro em produção sob *regime change*.
- **Não validado** para janelas de stress (COVID, eleições, crises).
- Não substitui análise fundamentalista nem técnica humana.

**Ação P0.** Criar `docs/model_cards/{ticker}_v{n}.md` para cada modelo treinado, no padrão [Model Cards (Mitchell et al., 2019)](https://arxiv.org/abs/1810.03993).

---

## 3. Explicabilidade — Camada 2: RAG (recuperação)

### 3.1 Estado atual

A função `retrieve` em [src/rag/retriever.py](src/rag/retriever.py) **já devolve** rank, distância L2, `doc_id`, `title` e texto do chunk (visto em [src/agent/tools.py:21-31](src/agent/tools.py#L21-L31)).

**Problema.** Em [src/serving/app.py:165-184](src/serving/app.py#L165-L184), `_query_with_rag` monta um `context` estruturado por `_build_context` ([app.py:67-104](src/serving/app.py#L67-L104)) — porém o `answer` é gerado a partir desse contexto **sem preservar vinculação por sentença**:

```python
results = _rerank_for_model_query(q, raw_results, top_k)
context = _build_context(results)
answer = generate_answer(q, context)
return {"query": q, "top_k": top_k, "context": context, "answer": answer}
```

O usuário recebe `context` e `answer` separados, sem saber **qual trecho do contexto sustenta cada afirmação da resposta** — embora os blocos `Fonte i:` em `_build_context` já tragam título e `fetched_at`.

### 3.2 Técnicas a implementar

| Técnica | Saída | Onde alterar |
|---------|-------|--------------|
| **Retornar `sources` estruturado** com `rank`, `score`, `doc_id`, `title`, `chunk_id` | Lista paralela ao `answer` | [app.py:165-184](src/serving/app.py#L165-L184) |
| **Citation in-line** no prompt do LLM (`[1]`, `[2]`) e parse pós-geração | Mapeamento `claim → chunk` | [src/rag/generator.py](src/rag/generator.py) (no prompt builder do path remoto/local) |
| **Faithfulness score** (ex.: RAGAS) | Métrica de quanto da `answer` é sustentada por `context` | Pipeline de avaliação + dashboard Grafana |
| **Confidence threshold por distância L2** | Recusar resposta se `min(distance) > τ` | [retriever.py](src/rag/retriever.py) |
| **Highlight do span recuperado** | Front exibe o trecho citado | [docs/index.html](docs/index.html) (página de demo) |

### 3.3 Resposta proposta para `/query`

```json
{
  "query": "Quais ações considerar em abril de 2026?",
  "answer": "Setor financeiro lidera em abril [1]. Renda fixa permanece atrativa [2].",
  "sources": [
    {
      "ref": 1,
      "doc_id": "boletim-2026-04",
      "title": "Análise Macroeconômica Abril 2026",
      "chunk_id": 3,
      "distance": 0.241,
      "snippet": "Setor financeiro lidera o desempenho mensal..."
    },
    {
      "ref": 2,
      "doc_id": "renda-fixa-2026",
      "title": "Cenário de Juros 2026",
      "chunk_id": 1,
      "distance": 0.298,
      "snippet": "Com Selic estável, renda fixa segue atrativa..."
    }
  ],
  "confidence": "medium",
  "min_distance": 0.241
}
```

### 3.4 Detecção de alucinação

- Após gerar `answer`, computar similaridade entre cada sentença e o conjunto de `sources`.
- Sentenças com baixa similaridade → marcar como `unsupported` ou suprimir.
- Métrica auditável persistida no MLflow.

---

## 4. Explicabilidade — Camada 3: agente ReAct e geração

### 4.1 Estado atual

O agente **já retorna `trace`** em [src/agent/react_agent.py:131-145](src/agent/react_agent.py#L131-L145), com `thought`, `action`, `action_input`, `observation` por step. Isso é **explicabilidade nativa do paradigma ReAct**. O endpoint `/agent` ainda adiciona um `trace` extra quando aplica o caminho rápido para queries de modelo ([app.py:286-318](src/serving/app.py#L286-L318)).

**Riscos da explicação atual.**
- O `thought` é gerado pelo próprio LLM remoto (vLLM/RunPod) e pode ser **post-hoc** (racionalização que não corresponde ao processo causal real).
- O `raw_output` ([react_agent.py:137](src/agent/react_agent.py#L137)) preserva a saída do modelo, mas pode conter system prompt vazado (ver RT-01 em [docs/RED_TEAM_REPORT.md](docs/RED_TEAM_REPORT.md#rt-01--prompt-injection-direta-com-vazamento-de-system-prompt)).
- Quando o agente cai no fallback simulado ([src/rag/generator.py:569+](src/rag/generator.py#L569)), o `trace` mostra raciocínio falso porque a resposta vem de regex hardcoded, **não** do LLM.
- Os fallbacks adicionados em [app.py:302-335](src/serving/app.py#L302-L335) substituem `answer` quando o agente retorna formato inválido — o `trace` é estendido com a justificativa, mas o usuário leigo pode não distinguir resposta agêntica de resposta RAG direta.

### 4.2 Melhorias propostas

| Melhoria | Implementação |
|----------|---------------|
| **Marcar resposta simulada / fallback** | Adicionar campo `generation_mode: "simulated" \| "rag_fast_path" \| "rag_fallback" \| "react"` em todas as respostas, distinguindo os caminhos de [app.py:286-335](src/serving/app.py#L286-L335) |
| **Sanitizar trace** antes de retornar | Remover system prompt, `HF_TOKEN` e `VLLM_API_KEY` se aparecerem no `raw_output` |
| **Tool-use auditing** | Cada chamada de tool gera registro estruturado (`tool`, `input_hash`, `output_summary`, `latency_ms`) persistido no **Langfuse** ([docker-compose.yaml:138-154](docker-compose.yaml#L138-L154)) |
| **Decisão revisável (LGPD Art. 20)** | Endpoint `POST /lgpd/review` que abre ticket vinculado a um `trace_id` para análise humana |
| **Determinismo opcional** | Permitir `temperature=0` em modo "auditoria" para reprodutibilidade do trace |

### 4.3 Limitações que devem ser comunicadas

- O `trace` mostra **o que o LLM disse que pensou**, não o que efetivamente computou.
- Múltiplas execuções com a mesma query podem gerar traces diferentes (não determinismo).
- A presença de `Final Answer` não garante que o agente realmente chegou a essa conclusão pelas tools listadas — pode ser confabulação.

---

## 5. Fairness — riscos identificados e métricas

### 5.1 Definições aplicáveis ao domínio

Não existe "fairness" universal — depende do dano que se quer evitar. No contexto deste projeto:

| Tipo de fairness | Pergunta correspondente | Aplicabilidade |
|------------------|-------------------------|----------------|
| **Group fairness** (paridade demográfica) | A acurácia da previsão é igual entre subgrupos de ativos? | **Sim** — subgrupos: setor (financeiro vs industrial), liquidez (large vs small cap), origem (BR vs internacional) |
| **Individual fairness** | Ativos similares recebem previsões similares? | Sim — relevante para small caps com histórico curto |
| **Counterfactual fairness** | A recomendação muda se o usuário declarar perfil diferente? | Não imediato (sistema não captura perfil) — relevante quando personalização entrar |
| **Procedural fairness** | O usuário consegue contestar a decisão? | **Sim** — exigido pelo Art. 20 LGPD |

### 5.2 Riscos concretos no projeto

#### R1 — Viés de cobertura nos modelos preditivos
Os modelos são treinados ticker-a-ticker via CLI `main(args)` em [train.py:262](src/models/train.py#L262). Para tickers com **histórico curto** (IPO recente) ou **baixa liquidez** (gaps no preço de fechamento), o pipeline de features ([src/features/feature_engineering.py](src/features/feature_engineering.py)) descarta amostras silenciosamente.

**Métrica.** Performance gap (MAPE) entre subgrupos: `mape_largecap` vs `mape_smallcap`, `mape_setor_financeiro` vs `mape_outros`.
**Limiar.** Razão entre maior e menor MAPE não deve exceder **1.5x**.

#### R2 — Viés de regime no treinamento
Split temporal 80/20 logado em [train.py:317](src/models/train.py#L317). Se o conjunto de treino concentra um regime macro (ex.: 2018–2023, taxa baixa) e o teste pega outro (2024+, Selic alta), a métrica de teste **subestima** o erro futuro.

**Métrica.** *Distribution shift* via PSI (Population Stability Index) entre treino e teste, e entre teste e produção.
**Limiar.** PSI > 0.2 → re-treinar; PSI > 0.1 → alerta.

#### R3 — Viés de conteúdo no RAG
Documentos ingeridos via `tool_fetch_news` ([src/rag/data_loader.py](src/rag/data_loader.py)) e `/ingest` podem refletir **fontes em português / mercado BR** desproporcionalmente. Queries sobre mercado internacional terão menos contexto e respostas piores.

**Métrica.** Cobertura por tópico: para um conjunto de queries-canário, medir `retrieval_recall@k` em cada categoria (BR vs internacional, renda fixa vs variável vs cripto).
**Limiar.** Recall@3 ≥ 0.7 em todas as categorias.

#### R4 — Viés de recomendação categórica
A função `_generate_simulated_answer` em [src/rag/generator.py:569+](src/rag/generator.py#L569) faz **dispatching por palavra-chave** que sempre direciona para perfis conservadores ("CDB, Tesouro Direto, debêntures"). Usuários cuja query menciona "criptomoeda" sempre recebem resposta de aversão a risco — mesmo que a query seja de um investidor profissional.

**Métrica.** Distribuição de classes de recomendação entre tipos de query. Se 95% das queries cripto recebem resposta "no máximo 5-10%", há rigidez (que pode ou não ser desejada — decidir explicitamente).
**Ação.** Documentar essa rigidez como **escolha de produto** ou substituí-la por geração condicionada ao perfil do usuário (com consentimento, ver [docs/LGPD_PLAN.md](docs/LGPD_PLAN.md)).

#### R5 — Viés do LLM upstream
Modelos servidos via `VLLM_BASE_URL` (default `qwen2.5-0.5b-awq` em [docker-compose.yaml:48](docker-compose.yaml#L48)) ou via fallback HF (`google/flan-t5-base` em [docker-compose.yaml:50](docker-compose.yaml#L50)) carregam vieses do corpus de treinamento — sub-representação de ativos brasileiros, gírias financeiras em PT-BR, gênero/raça em narrativas.

**Métrica.** Bias eval com benchmarks como [HolisticBias](https://github.com/facebookresearch/ResponsibleNLP), traduzido/adaptado para PT-BR.
**Ação.** Documentar no model card do RAG quais modelos foram avaliados e onde falham.

### 5.3 Métricas a instrumentar

| Métrica | Onde calcular | Frequência |
|---------|---------------|------------|
| MAPE por subgrupo de ticker (setor, liquidez) | Pipeline de avaliação batch (DVC) | Por release de modelo |
| PSI treino↔teste↔produção | Pipeline de monitoramento + Prometheus | Diário |
| Retrieval recall@k por categoria | Suite de queries-canário | Por mudança no índice |
| Faithfulness score (RAGAS) | Avaliação pós-geração | Por mudança no prompt |
| Distribuição de classes de resposta | Logs de produção (Langfuse) | Semanal |
| Latência média e p95 por step do agente | Trace Langfuse + counter Prometheus (`:8001`) | Contínuo |

---

## 6. Plano de implementação

### Sprint 1 (P0 — fundação)
- [ ] Criar Model Card para cada modelo em `docs/model_cards/` (complementa [docs/MODEL_CARD.md](docs/MODEL_CARD.md)).
- [ ] Adicionar campo `sources` na resposta de `/query` com citação estruturada.
- [ ] Marcar `generation_mode` em `/query` e `/agent` (distinguindo `react`, `rag_fast_path`, `rag_fallback`, `simulated`).
- [ ] Sanitizar `trace` do agente (remover system prompt, `HF_TOKEN`, `VLLM_API_KEY`).
- [ ] Endpoint `POST /lgpd/review` para revisão humana (Art. 20).

### Sprint 2 (P1 — XAI dos modelos)
- [ ] SHAP / Integrated Gradients no PyTorch MLP via Captum.
- [ ] Permutation importance no Sklearn MLP.
- [ ] Confidence interval via MC Dropout no Keras LSTM.
- [ ] Adicionar `top_features` e `confidence_interval_95` na resposta de inferência.

### Sprint 3 (P1 — fairness instrumentado)
- [ ] Suite de avaliação por subgrupo (setor, liquidez, regime macro).
- [ ] PSI computado em pipeline DVC ([dvc.yaml](dvc.yaml)).
- [ ] Suite de queries-canário e medição de recall@k por categoria.
- [ ] RAGAS faithfulness score automatizado.
- [ ] Dashboard de fairness em Grafana ([configs/grafana/](configs/grafana/)).

### Sprint 4 (P2 — governança contínua)
- [ ] Bias eval do LLM com benchmark PT-BR.
- [ ] Auditoria humana periódica de outputs de recomendação.
- [ ] Re-treino automático quando PSI > 0.2.

---

## 7. Métricas e dashboard

Sugestão de painel exposto em **Grafana** (`:3001`, ver [docker-compose.yaml:115-135](docker-compose.yaml#L115-L135)) com datasource Prometheus + MLflow, complementado pelo endpoint `GET /metrics/fairness` (interno, autenticado):

```
┌──────────────────────────────────────────────────────────┐
│ MODEL FAIRNESS — última atualização: 2026-04-28 03:00    │
├──────────────────────────────────────────────────────────┤
│ MAPE (overall):              4.2%                        │
│ MAPE (large caps):           3.8%   ┐                    │
│ MAPE (small caps):           5.9%   │ ratio = 1.55  ⚠️    │
│ MAPE (setor financeiro):     3.1%                        │
│ MAPE (setor industrial):     4.7%                        │
├──────────────────────────────────────────────────────────┤
│ PSI treino → produção (30d): 0.08   ✓                    │
│ Drift alert:                 nenhum                      │
├──────────────────────────────────────────────────────────┤
│ RAG faithfulness (RAGAS):    0.81   ✓                    │
│ Retrieval recall@3 BR:       0.84   ✓                    │
│ Retrieval recall@3 INTL:     0.61   ⚠️                    │
└──────────────────────────────────────────────────────────┘
```

Os limiares (1.5x ratio, PSI 0.2, recall 0.7) ficam em [config/](config/) versionados (ex.: [config/monitoring_config.yaml](config/monitoring_config.yaml)).

---

## 8. Governança e ciclo de revisão

- **Revisão trimestral** dos Model Cards e do plano de fairness.
- **Disclaimer obrigatório** em toda resposta financeira (ver [docs/RED_TEAM_REPORT.md](docs/RED_TEAM_REPORT.md#rt-02--data-poisoning-via-ingest-ingest_mlflow-e-fetch_news) §RT-02 — disclaimer CVM 39/2021).
- **Direito à revisão humana (LGPD Art. 20)** acessível em até **15 dias úteis**, conforme [docs/LGPD_PLAN.md](docs/LGPD_PLAN.md#5-direitos-dos-titulares-art-18--como-atender) §5.
- **Auditoria externa** anual com foco em viés sistemático.
- **Comitê de IA Responsável** (mesmo que pequeno): Encarregado LGPD + um eng. ML + um stakeholder de produto. Aprova mudanças de prompt do agente, novos modelos e novas categorias de recomendação.
- **Versionamento de prompts**: registrar [react_agent.py:59-94](src/agent/react_agent.py#L59-L94) sob versionamento semântico — mudanças exigem revisão.

---

## 9. Referências

### Explicabilidade
- [Mitchell et al., *Model Cards for Model Reporting*, 2019](https://arxiv.org/abs/1810.03993)
- [Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions* (SHAP), 2017](https://arxiv.org/abs/1705.07874)
- [Sundararajan et al., *Axiomatic Attribution for Deep Networks* (Integrated Gradients), 2017](https://arxiv.org/abs/1703.01365)
- [Captum (PyTorch interpretability)](https://captum.ai/)
- [RAGAS — RAG Assessment](https://github.com/explodinggradients/ragas)
- [Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, 2022](https://arxiv.org/abs/2210.03629)

### Fairness
- [Barocas, Hardt, Narayanan — *Fairness and Machine Learning*, 2023](https://fairmlbook.org/)
- [IBM AI Fairness 360](https://aif360.res.ibm.com/)
- [Microsoft Fairlearn](https://fairlearn.org/)
- [HolisticBias — Meta Responsible NLP](https://github.com/facebookresearch/ResponsibleNLP)
- [PSI — Population Stability Index (referência de risco de crédito)](https://scholarworks.wmich.edu/dissertations/3208/)

### Governança
- [LGPD — Art. 20 (decisões automatizadas)](https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm#art20)
- [NIST AI Risk Management Framework — Trustworthy & Responsible AI](https://www.nist.gov/itl/ai-risk-management-framework)
- [EU AI Act — Title III (sistemas de alto risco)](https://artificialintelligenceact.eu/)
- Documentação interna: [docs/OWASP.md](docs/OWASP.md), [docs/RED_TEAM_REPORT.md](docs/RED_TEAM_REPORT.md), [docs/LGPD_PLAN.md](docs/LGPD_PLAN.md), [docs/MODEL_CARD.md](docs/MODEL_CARD.md)
