# Explicabilidade e Fairness — Datathon MLET (Grupo 05)

Documento que **operacionaliza** os requisitos de explicabilidade (XAI) e *fairness* sobre o sistema atual, composto por três camadas com perfis de risco distintos:

1. **Modelos preditivos de regressão** sobre série temporal de preços (PyTorch MLP, Sklearn MLP, Keras LSTM) — ver [src/models/train.py](src/models/train.py).
2. **RAG** com índice FAISS sobre documentos financeiros — ver [src/rag/](src/rag/).
3. **Agente ReAct** que orquestra ferramentas e gera recomendações — ver [src/agent/react_agent.py](src/agent/react_agent.py).

Todas as camadas produzem ou influenciam **recomendações financeiras**, o que ativa o **Art. 20 da LGPD** (direito à revisão de decisões automatizadas) e o **Art. 6º, IX** (princípio da não discriminação). Este documento é complementar a [docs/OWASP.md](docs/OWASP.md), [docs/RED_TEAMING.md](docs/RED_TEAMING.md) e [docs/LGPD_PLAN.md](docs/LGPD_PLAN.md).

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
| Decisão automatizada com efeito financeiro | Recomendações de ações/alocação | [src/rag/generator.py:148-161](src/rag/generator.py#L148-L161), [src/agent/react_agent.py:60-62](src/agent/react_agent.py#L60-L62) |
| Modelo de séries temporais sem feature importance | MLP/LSTM tratam preço como única entrada — usuário não sabe **por que** previu X | [src/models/train.py:43-54](src/models/train.py#L43-L54), [src/models/train.py:255-262](src/models/train.py#L255-L262) |
| Recuperação RAG sem citação visível na resposta | `/query` retorna `answer` desacoplado das fontes | [src/serving/app.py:50-58](src/serving/app.py#L50-L58) |
| Agente ReAct com trace exposto, mas sem garantia de fidelidade | Trace pode ser *post-hoc*, não causal | [src/agent/react_agent.py:97-163](src/agent/react_agent.py#L97-L163) |
| Viés potencial em recomendações | Concentração em ativos *blue-chip* / mercado brasileiro / horário comercial dos dados de treino | [src/rag/generator.py:148-171](src/rag/generator.py#L148-L171) |
| Decisões automatizadas (LGPD Art. 20) | Direito à revisão por pessoa natural | — não implementado |

---

## 2. Explicabilidade — Camada 1: modelos preditivos

### 2.1 Estado atual

Três modelos treinados sobre janela univariada de preços `Close` ([src/models/train.py:116-131](src/models/train.py#L116-L131)):

| Modelo | Arquitetura | Saída interpretável hoje? |
|--------|-------------|---------------------------|
| PyTorch MLP | 3 camadas densas (input → 64 → 32 → 1) | Não |
| Sklearn MLPRegressor | hidden_layer_sizes via config | Não |
| Keras LSTM | LSTM(50) → Dropout → LSTM(50) → Dropout → Dense(1) | Não |
| Baseline | `y_pred = X_test[:, -1]` (último valor) | Sim (trivial) |

**Métricas atuais.** MAE, RMSE, MAPE registradas no MLflow ([train.py:79-91](src/models/train.py#L79-L91)). Métricas de **erro**, mas **não de explicabilidade**.

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

A função `retrieve` em [src/rag/retriever.py](src/rag/retriever.py) **já devolve** rank, distância L2, `doc_id`, `title` e texto do chunk (visto em [src/agent/tools.py:20-30](src/agent/tools.py#L20-L30)).

**Problema.** Em [src/serving/app.py:55-58](src/serving/app.py#L55-L58), `/query` concatena os chunks em uma string `context` e o LLM gera `answer` **sem preservar a vinculação**:

```python
context = " ".join([r["text"] for r in results])
answer = generate_answer(q, context)
return {"query": q, "top_k": top_k, "context": context, "answer": answer}
```

O usuário recebe `context` e `answer` separados, sem saber **qual trecho do contexto sustenta cada afirmação da resposta**.

### 3.2 Técnicas a implementar

| Técnica | Saída | Onde alterar |
|---------|-------|--------------|
| **Retornar `sources` estruturado** com `rank`, `score`, `doc_id`, `title`, `chunk_id` | Lista paralela ao `answer` | [app.py:50-58](src/serving/app.py#L50-L58) |
| **Citation in-line** no prompt do LLM (`[1]`, `[2]`) e parse pós-geração | Mapeamento `claim → chunk` | [generator.py:120](src/rag/generator.py#L120) |
| **Faithfulness score** (ex.: RAGAS) | Métrica de quanto da `answer` é sustentada por `context` | Pipeline de avaliação |
| **Confidence threshold por distância L2** | Recusar resposta se `min(distance) > τ` | [retriever.py](src/rag/retriever.py) |
| **Highlight do span recuperado** | Front exibe o trecho citado | `api_test.html` |

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

O agente **já retorna `trace`** em [src/agent/react_agent.py:135-139](src/agent/react_agent.py#L135-L139), com `thought`, `action`, `action_input`, `observation` por step. Isso é **explicabilidade nativa do paradigma ReAct**.

**Riscos da explicação atual.**
- O `thought` é gerado pelo próprio LLM e pode ser **post-hoc** (racionalização que não corresponde ao processo causal real).
- O `raw_output` ([react_agent.py:130](src/agent/react_agent.py#L130)) preserva a saída do modelo, mas pode conter system prompt vazado (ver RT-01 em [docs/RED_TEAMING.md](docs/RED_TEAMING.md#rt-01--prompt-injection-direta-com-vazamento-de-system-prompt)).
- Quando o agente cai no fallback simulado ([generator.py:137-171](src/rag/generator.py#L137-L171)), o `trace` mostra raciocínio falso porque a resposta vem de regex hardcoded, **não** do LLM.

### 4.2 Melhorias propostas

| Melhoria | Implementação |
|----------|---------------|
| **Marcar resposta simulada** | Adicionar campo `generation_mode: "simulated"` e badge `[DEMO]` em respostas geradas por `_generate_simulated_answer` |
| **Sanitizar trace** antes de retornar | Remover system prompt e `HF_TOKEN` se aparecerem no `raw_output` |
| **Tool-use auditing** | Cada chamada de tool gera registro estruturado (`tool`, `input_hash`, `output_summary`, `latency_ms`) persistido no MLflow |
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
Os modelos são treinados ticker-a-ticker via `--ticker` em [train.py:307](src/models/train.py#L307). Para tickers com **histórico curto** (IPO recente) ou **baixa liquidez** (gaps no preço de fechamento), a janela em [train.py:116-131](src/models/train.py#L116-L131) descarta amostras silenciosamente.

**Métrica.** Performance gap (MAPE) entre subgrupos: `mape_largecap` vs `mape_smallcap`, `mape_setor_financeiro` vs `mape_outros`.
**Limiar.** Razão entre maior e menor MAPE não deve exceder **1.5x**.

#### R2 — Viés de regime no treinamento
Split temporal 80/20 em [train.py:169-171](src/models/train.py#L169-L171). Se o conjunto de treino concentra um regime macro (ex.: 2018–2023, taxa baixa) e o teste pega outro (2024+, Selic alta), a métrica de teste **subestima** o erro futuro.

**Métrica.** *Distribution shift* via PSI (Population Stability Index) entre treino e teste, e entre teste e produção.
**Limiar.** PSI > 0.2 → re-treinar; PSI > 0.1 → alerta.

#### R3 — Viés de conteúdo no RAG
Documentos ingeridos via `tool_fetch_news` ([src/rag/data_loader.py](src/rag/data_loader.py)) e `/ingest` podem refletir **fontes em português / mercado BR** desproporcionalmente. Queries sobre mercado internacional terão menos contexto e respostas piores.

**Métrica.** Cobertura por tópico: para um conjunto de queries-canário, medir `retrieval_recall@k` em cada categoria (BR vs internacional, renda fixa vs variável vs cripto).
**Limiar.** Recall@3 ≥ 0.7 em todas as categorias.

#### R4 — Viés de recomendação categórica
A função `_generate_simulated_answer` em [src/rag/generator.py:148-161](src/rag/generator.py#L148-L161) faz **dispatching por palavra-chave** que sempre direciona para perfis conservadores ("CDB, Tesouro Direto, debêntures"). Usuários cuja query menciona "criptomoeda" sempre recebem resposta de aversão a risco — mesmo que a query seja de um investidor profissional.

**Métrica.** Distribuição de classes de recomendação entre tipos de query. Se 95% das queries cripto recebem resposta "no máximo 5-10%", há rigidez (que pode ou não ser desejada — decidir explicitamente).
**Ação.** Documentar essa rigidez como **escolha de produto** ou substituí-la por geração condicionada ao perfil do usuário (com consentimento, ver [docs/LGPD_PLAN.md](docs/LGPD_PLAN.md)).

#### R5 — Viés do LLM upstream
Modelos como `Qwen/Qwen2.5-0.5B-Instruct-AWQ` (ver [README.md:124](README.md#L124)) ou `facebook/opt-1.3b` ([generator.py:14](src/rag/generator.py#L14)) carregam vieses do corpus de treinamento — sub-representação de ativos brasileiros, gírias financeiras em PT-BR, gênero/raça em narrativas.

**Métrica.** Bias eval com benchmarks como [HolisticBias](https://github.com/facebookresearch/ResponsibleNLP), traduzido/adaptado para PT-BR.
**Ação.** Documentar no model card do RAG quais modelos foram avaliados e onde falham.

### 5.3 Métricas a instrumentar

| Métrica | Onde calcular | Frequência |
|---------|---------------|------------|
| MAPE por subgrupo de ticker (setor, liquidez) | Pipeline de avaliação batch | Por release de modelo |
| PSI treino↔teste↔produção | Pipeline de monitoramento | Diário |
| Retrieval recall@k por categoria | Suite de queries-canário | Por mudança no índice |
| Faithfulness score (RAGAS) | Avaliação pós-geração | Por mudança no prompt |
| Distribuição de classes de resposta | Logs de produção | Semanal |
| Latência média e p95 por step do agente | Trace MLflow | Contínuo |

---

## 6. Plano de implementação

### Sprint 1 (P0 — fundação)
- [ ] Criar Model Card para cada modelo em `docs/model_cards/`.
- [ ] Adicionar campo `sources` na resposta de `/query` com citação estruturada.
- [ ] Marcar `generation_mode` em `/query` e `/agent` quando resposta é simulada.
- [ ] Sanitizar `trace` do agente (remover system prompt e tokens).
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
- [ ] Dashboard de fairness exposto.

### Sprint 4 (P2 — governança contínua)
- [ ] Bias eval do LLM com benchmark PT-BR.
- [ ] Auditoria humana periódica de outputs de recomendação.
- [ ] Re-treino automático quando PSI > 0.2.

---

## 7. Métricas e dashboard

Sugestão de painel exposto em endpoint `GET /metrics/fairness` (interno, autenticado), agregando do MLflow:

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

Os limiares (1.5x ratio, PSI 0.2, recall 0.7) ficam em [config/](config/) versionados.

---

## 8. Governança e ciclo de revisão

- **Revisão trimestral** dos Model Cards e do plano de fairness.
- **Disclaimer obrigatório** em toda resposta financeira (ver [docs/RED_TEAMING.md](docs/RED_TEAMING.md#rt-04--manipulação-de-recomendação-financeira-pump--dump-assistido-por-ia) §RT-04).
- **Direito à revisão humana (LGPD Art. 20)** acessível em até **15 dias úteis**, conforme [docs/LGPD_PLAN.md](docs/LGPD_PLAN.md#5-direitos-dos-titulares-art-18--como-atender) §5.
- **Auditoria externa** anual com foco em viés sistemático.
- **Comitê de IA Responsável** (mesmo que pequeno): Encarregado LGPD + um eng. ML + um stakeholder de produto. Aprova mudanças de prompt do agente, novos modelos e novas categorias de recomendação.
- **Versionamento de prompts**: registrar [react_agent.py:60-87](src/agent/react_agent.py#L60-L87) sob versionamento semântico — mudanças exigem revisão.

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
- Documentação interna: [docs/OWASP.md](docs/OWASP.md), [docs/RED_TEAMING.md](docs/RED_TEAMING.md), [docs/LGPD_PLAN.md](docs/LGPD_PLAN.md)
