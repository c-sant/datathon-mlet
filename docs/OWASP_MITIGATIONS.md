# OWASP — 5 Mitigações Implementadas no Projeto

Este documento descreve as **5 mitigações concretas** já presentes no código do **Datathon MLET** — um sistema de **RAG + Agente ReAct** servido via **FastAPI**, com geração delegada a um **LLM remoto (vLLM/RunPod ou BentoML)** com fallback Hugging Face local, índice vetorial **FAISS** e *stack* de observabilidade **Prometheus + Grafana + Langfuse**. Cada mitigação é mapeada a uma ameaça do **[OWASP Top 10 for LLM Applications (2025)](https://genai.owasp.org/llm-top-10/)** ou da **[OWASP API Security Top 10 (2023)](https://owasp.org/API-Security/editions/2023/en/0x11-t10/)**, com o controle aplicado, os pontos do código onde a mitigação vive e as limitações conhecidas.

---

## 1. Detecção de Prompt Injection na entrada do usuário (LLM01:2025)

**Controle.** `InputGuardrail` baseado em regex bloqueia padrões conhecidos de injeção em **inglês e português** antes que a query chegue ao LLM (vLLM remoto ou HF local) ou ao agente ReAct.

**Onde.**
- [src/security/guardrails.py:14-32](src/security/guardrails.py#L14-L32) — 17 padrões regex incluindo `ignore previous instructions`, `DAN mode`, `developer mode`, `desconsidere as instruções acima`, `esqueça as regras anteriores`, `aja como se fosse o desenvolvedor`, `modo desenvolvedor ativado`, `[override]`, `system:`, etc., além de limite de 4096 caracteres.
- [src/serving/app.py:80-89](src/serving/app.py#L80-L89) — `_validate_user_query()` normaliza texto, chama `validate()`, e levanta `HTTPException(400)` quando o input é bloqueado.
- Aplicado em `GET /query` ([app.py:306](src/serving/app.py#L306)) e `POST /agent` ([app.py:332](src/serving/app.py#L332)).
- Fork [generator/serving/app.py:56-62](generator/serving/app.py#L56-L62) — `field_validator` no `Document.text` rejeita prompt injection no payload do `/ingest` antes da indexação.


---

## 2. Sanitização de PII na saída do LLM (LLM02:2025)

**Controle.** `OutputGuardrail` com **Microsoft Presidio** detecta e mascara PII antes de devolver respostas ao cliente, com **fallback regex local** equivalente caso o Presidio falhe ao iniciar (modelo spaCy ausente, idioma não suportado).

**Onde.**
- [src/security/guardrails.py:51-165](src/security/guardrails.py#L51-L165) — `OutputGuardrail` cobre `PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `BR_CPF`, `BR_CNPJ`, `CREDIT_CARD`, `IBAN_CODE`, `IP_ADDRESS`.
- Fallback regex local em [guardrails.py:54-60,82-100](src/security/guardrails.py#L54-L60) — padrões `EMAIL_PATTERN`, `PHONE_PATTERN`, `CPF_PATTERN`, `CNPJ_PATTERN`, `CREDIT_CARD_PATTERN`, `IBAN_PATTERN`, `IP_ADDRESS_PATTERN` com substituição por placeholders (`<EMAIL_ADDRESS>`, `<BR_CPF>`, etc.).
- Fallback de idioma EN em [guardrails.py:148-165](src/security/guardrails.py#L148-L165) — quando o pacote PT do Presidio não está disponível.
- Aplicado **recursivamente** em `_sanitize_public_value` ([src/serving/app.py:92-104](src/serving/app.py#L92-L104)), que percorre dicts/listas e sanitiza todos os campos string das respostas (`context`, `answer`, `trace`).
- Sanitização adicional na saída do gerador local: [src/rag/generator.py:256](src/rag/generator.py#L256) — `_clean_generated_answer` chama `OutputGuardrail().sanitize()` antes de retornar.
- Sanitização também na **entrada** do `/ingest` no fork: [generator/serving/app.py:62](generator/serving/app.py#L62) remove PII do texto antes da indexação no FAISS.


---

## 3. Restrição de ferramentas e raciocínio do agente (LLM06:2025)

**Controle.** O agente ReAct opera com um **registry fechado de ferramentas** (`TOOL_MAP`) e tem o número máximo de iterações (`max_steps`) controlado por configuração — o agente não pode inventar ações nem entrar em loop infinito.

**Onde.**
- [src/agent/tools.py:90-117](src/agent/tools.py#L90-L117) — `TOOLS` define exatamente 3 ferramentas (`search_documents`, `fetch_news`, `summarize_context`); `TOOL_MAP = {tool.name: tool for tool in TOOLS}` é o único mapa consultado pelo agente.
- [src/agent/react_agent.py:97-101](src/agent/react_agent.py#L97-L101) — `_execute_tool` retorna `"Ferramenta desconhecida: {action}"` se o LLM tentar acionar algo fora do registry.
- [src/agent/react_agent.py:113](src/agent/react_agent.py#L113) — loop principal limitado por `max_steps` lido de [config/model_config.yaml](config/model_config.yaml).
- [src/agent/react_agent.py:59-94](src/agent/react_agent.py#L59-L94) — system prompt fixa formato ReAct estrito (`Thought:`/`Action:`/`Action Input:`/`Final Answer:`), reduzindo superfície de manipulação.


---

## 4. Validação e sanitização na ingestão de documentos (LLM04:2025)

**Controle.** No fork `generator/serving/app.py`, todo documento enviado ao `/ingest` passa por **validação de Pydantic + guardrail de segurança** antes de ser indexado no FAISS — combina barreira contra prompt injection indireta com remoção de PII na entrada.

**Onde.**
- [generator/serving/app.py:51-62](generator/serving/app.py#L51-L62) — modelo `Document` define `field_validator("text")` que: (a) chama `InputGuardrail.validate()` desempacotando `is_ok, reason`; (b) levanta `ValueError(f"Texto inválido: {reason}")` quando o input falha (Pydantic devolve `HTTP 422`); (c) aplica `OutputGuardrail.sanitize()` no texto antes de armazená-lo.
- [generator/serving/app.py:66](generator/serving/app.py#L66) — `IngestRequest.docs: list[Document]` força o validador a rodar em **todos** os documentos do payload.
- [generator/serving/app.py:213-233](generator/serving/app.py#L213-L233) — endpoint `/ingest` só recebe documentos que já passaram pelos validadores; `try/except ValueError` traduz falhas de validação em `HTTP 400`.


---

## 5. Stack de observabilidade para detecção de abuso (API1:2023 / A05:2021 — defesa em profundidade)

**Controle.** Stack **Prometheus + Grafana + Langfuse** instrumentado via Docker Compose permite detectar padrões anômalos de uso (picos de requisições, erros recorrentes, latência atípica do vLLM) que sinalizam exploração das demais ameaças.

**Onde.**
- [docker-compose.yaml:97-113](docker-compose.yaml#L97-L113) — Prometheus coletando métricas (porta `9090`, retenção 30d).
- [docker-compose.yaml:116-135](docker-compose.yaml#L116-L135) — Grafana servindo dashboards (porta `3001`).
- [docker-compose.yaml:138-154](docker-compose.yaml#L138-L154) — Langfuse com PostgreSQL dedicado para tracing de chamadas LLM (porta `3000`).
- [docker-compose.yaml:73,81](docker-compose.yaml#L73) — `rag-app` expõe `PROMETHEUS_METRICS_PORT=8001` para scrape.
- [config/monitoring_config.yaml](config/monitoring_config.yaml) e `configs/prometheus.yml` ([docker-compose.yaml:103](docker-compose.yaml#L103)) parametrizam métricas e alvos.


---

## Referências

- [src/security/guardrails.py](src/security/guardrails.py) — implementação dos guardrails de input e output.
- [src/serving/app.py](src/serving/app.py) — caminho canônico da API com `_validate_user_query` e `_sanitize_public_value`.
- [generator/serving/app.py](generator/serving/app.py) — fork com `field_validator` aplicado na ingestão.
- [docker-compose.yaml](docker-compose.yaml) — stack de observabilidade.
- [OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/llm-top-10/)
- [OWASP API Security Top 10 (2023)](https://owasp.org/API-Security/editions/2023/en/0x11-t10/)
