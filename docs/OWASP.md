# OWASP — Análise de Ameaças do Projeto

Documento de mapeamento de ameaças de segurança aplicáveis ao **Datathon MLET — Grupo 05**, um sistema de **RAG + Agente ReAct** servido via **FastAPI**, com geração de texto delegada a um **LLM remoto (vLLM/RunPod ou BentoML)** com fallback Hugging Face local, índice vetorial **FAISS** e *stack* de observabilidade **Prometheus + Grafana + Langfuse**.

A análise utiliza como referência principal a **[OWASP Top 10 for LLM Applications (2025)](https://genai.owasp.org/llm-top-10/)**, complementada com itens da **OWASP API Security Top 10** quando relevantes para a superfície HTTP exposta pela aplicação.

---

## 1. Contexto do sistema sob análise

| Componente | Localização | Exposição |
|------------|-------------|-----------|
| API HTTP | [src/serving/app.py](src/serving/app.py) | Endpoints públicos `/ingest`, `/ingest_mlflow`, `/query`, `/agent` |
| Agente ReAct | [src/agent/react_agent.py](src/agent/react_agent.py) | Executa ferramentas (`search_documents`, `fetch_news`, `summarize_context`) |
| Pipeline RAG | [src/rag/embedding.py](src/rag/embedding.py), [src/rag/generator.py](src/rag/generator.py) | Embeddings (SentenceTransformers) + FAISS + LLM remoto/local |
| Loader MLflow | [src/rag/mlflow_loader.py](src/rag/mlflow_loader.py) | Lê métricas/parâmetros do pipeline DVC e injeta no índice via `/ingest_mlflow` |
| Guardrails | [src/security/guardrails.py](src/security/guardrails.py) | Detecção de prompt injection (regex) + PII em output (Presidio) |
| Geração LLM | vLLM remoto (`VLLM_BASE_URL` — RunPod), BentoML (`RAG_GENERATOR_URL`) ou HF local (`RAG_MODEL`) | Configurável via variável de ambiente |
| Observabilidade | [docker-compose.yaml:96-172](docker-compose.yaml#L96-L172) | Prometheus (`:9090`), Grafana (`:3001`), Langfuse (`:3000`) |

Superfície de ataque principal: **entradas de texto não confiáveis enviadas pelos endpoints HTTP**, que alimentam o LLM remoto, o índice vetorial e as chamadas de ferramentas do agente.

---

## 2. Resumo executivo das ameaças mapeadas

| # | ID OWASP | Ameaça | Severidade | Status atual |
|---|----------|--------|------------|--------------|
| 1 | LLM01:2025 | Prompt Injection (direta e indireta via RAG/`fetch_news`) | **Alta** | Mitigação parcial (regex em [guardrails.py:15-28](src/security/guardrails.py#L15-L28)) — **não integrado em `app.py`** |
| 2 | LLM02:2025 | Sensitive Information Disclosure (PII + segredos HF/vLLM) | **Alta** | Mitigação parcial (Presidio em output) — **não integrado em `app.py`** |
| 3 | LLM04:2025 | Data and Model Poisoning via `/ingest`, `/ingest_mlflow` e `fetch_news` | **Alta** | **Sem mitigação** |
| 4 | LLM06:2025 | Excessive Agency do agente ReAct (efeitos colaterais de tools) | **Média** | Mitigação parcial (`TOOL_MAP` fechado, `max_steps`) |
| 5 | API1:2023 + A05:2021 | Broken Access Control + Security Misconfiguration (sem auth, CORS `*`, defaults de Grafana/Langfuse) | **Crítica** | **Sem mitigação** |

---

## 3. Ameaças detalhadas

### 3.1 LLM01:2025 — Prompt Injection

**Descrição.** Um atacante injeta instruções que sobrescrevem o *system prompt* do agente, forçando o LLM a ignorar restrições, vazar o prompt, executar ferramentas indevidas ou produzir respostas maliciosas. Pode ser **direta** (no campo `q` de `/query` ou `query` de `/agent`) ou **indireta** (via documentos enviados em `/ingest`, métricas indexadas via `/ingest_mlflow` ou notícias coletadas pela tool `fetch_news`).

**Onde no projeto.**
- Entrada direta: [app.py:249](src/serving/app.py#L249) (`/query`) e [app.py:275](src/serving/app.py#L275) (`/agent`).
- Entrada indireta: [embedding.py:85-206](src/rag/embedding.py#L85-L206) (qualquer texto ingerido vira contexto do LLM em consultas futuras) e [tools.py:56-65](src/agent/tools.py#L56-L65) (`fetch_news` injeta conteúdo externo no índice global).
- Prompt do agente embutido no código: [react_agent.py:59-94](src/agent/react_agent.py#L59-L94).

**Impacto.** Vazamento do system prompt, execução não autorizada de tools (`fetch_news` para forçar coleta externa, `search_documents` para enumeração), exfiltração de contexto de outros usuários (índice FAISS é **global e compartilhado**), respostas tendenciosas/maliciosas enviadas ao endpoint vLLM remoto (RunPod).

**Mitigação atual.** [guardrails.py:15-53](src/security/guardrails.py#L15-L53) bloqueia ~12 padrões regex em inglês ("ignore previous instructions", "DAN mode", etc.). **Limitação crítica:** o `InputGuardrail` **não é importado nem chamado** em [src/serving/app.py](src/serving/app.py) — está implementado mas desconectado do fluxo de requisição. Padrões em português também não são cobertos.

**Mitigações recomendadas.**
1. **Integrar `InputGuardrail.validate()` em `/query` e `/agent`** antes de chegar ao LLM remoto.
2. Aplicar guardrail também sobre documentos em `/ingest` e `/ingest_mlflow` (injeção indireta).
3. Adicionar padrões em português ao regex ("ignore as instruções acima", "esqueça as regras", "modo desenvolvedor").
4. Usar separadores estruturados/delimitadores no prompt do agente e/ou *spotlighting* para distinguir instrução de dados.
5. Avaliar classificadores ML para detecção de injection (ex.: `protectai/deberta-v3-base-prompt-injection-v2`).
6. Não confiar na saída do LLM para decisões privilegiadas — sempre validar antes de executar tools.

---

### 3.2 LLM02:2025 — Sensitive Information Disclosure

**Descrição.** O LLM pode reproduzir PII presente nos documentos ingeridos, vazar tokens/segredos do ambiente, ou regurgitar trechos de seu corpus de treino contendo dados sensíveis. Adicionalmente, com a integração ao **vLLM remoto na RunPod**, todo conteúdo de `query` e `context` é enviado para fora da fronteira da aplicação.

**Onde no projeto.**
- Variáveis sensíveis: `HF_TOKEN`, `HUGGINGFACEHUB_API_TOKEN` em [generator.py:25-29](src/rag/generator.py#L25-L29) e [embedding.py:25-29](src/rag/embedding.py#L25-L29); `VLLM_API_KEY`/`OPENAI_API_KEY` em [generator.py:14](src/rag/generator.py#L14). Se acidentalmente impressas em log de erro ou colocadas no contexto, podem ser ecoadas.
- Documentos ingeridos via `/ingest` podem conter CPF, e-mail, telefone, nome de pessoas — e são devolvidos como `context` em `/query` ([app.py:184](src/serving/app.py#L184)) sem sanitização.
- Encaminhamento ao endpoint remoto: chamadas a `VLLM_BASE_URL` em [generator.py](src/rag/generator.py) enviam *prompt completo* ao serviço externo (RunPod); qualquer PII no contexto vaza para fora da aplicação.

**Impacto.** Vazamento de PII (violação **LGPD**), exposição de credenciais HF/vLLM (uso indevido de cota), exposição de dados confidenciais corporativos para serviço de inferência terceirizado.

**Mitigação atual.** [guardrails.py:56-92](src/security/guardrails.py#L56-L92) — `OutputGuardrail` com Presidio detectando `PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `BR_CPF`. **Limitações:** (a) não é importado nem chamado em [src/serving/app.py](src/serving/app.py); (b) `/query` retorna o `context` bruto sem sanitização ([app.py:184](src/serving/app.py#L184)); (c) cobertura de PII brasileiro restrita (faltam RG, CNPJ, dados bancários).

**Mitigações recomendadas.**
1. Aplicar `OutputGuardrail.sanitize()` em **toda** resposta de `/query`, `/agent` — incluindo o campo `context`.
2. **Sanitizar antes do envio remoto**: aplicar PII scrubbing no prompt antes de chamar `VLLM_BASE_URL`, garantindo que dados sensíveis não saiam da fronteira da aplicação.
3. Adicionar entidades adicionais ao Presidio: `BR_CNPJ`, `CREDIT_CARD`, `IBAN_CODE`, `IP_ADDRESS`.
4. Sanitizar PII também na **entrada** (em `/ingest`) — não armazenar PII no índice.
5. Garantir que `HF_TOKEN`, `VLLM_API_KEY` **nunca** entrem no prompt; revisar logs (`print(f"...{exc}...")` em [embedding.py:193](src/rag/embedding.py#L193)) para evitar vazamento.
6. Política explícita no system prompt: "nunca repita números, e-mails ou nomes próprios extraídos do contexto".
7. Avaliar contratualmente com a RunPod retenção/uso dos prompts enviados para o endpoint vLLM.

---

### 3.3 LLM04:2025 — Data and Model Poisoning

**Descrição.** Os endpoints `/ingest` e `/ingest_mlflow` aceitam adição ao índice FAISS global **sem autenticação**. `/ingest` permite ainda `overwrite=True` por padrão ([app.py:45](src/serving/app.py#L45)), permitindo substituição completa da base de conhecimento. A tool `fetch_news` baixa notícias de fontes externas e as injeta no índice global durante a execução do agente. Um atacante pode plantar documentos com instruções maliciosas (injeção indireta), desinformação financeira ou dados falsos que serão recuperados como "contexto autoritativo" para todos os usuários subsequentes.

**Onde no projeto.**
- [app.py:187-212](src/serving/app.py#L187-L212) — `/ingest` aceita `IngestRequest` sem autenticação, com `overwrite=True` por padrão.
- [app.py:215-241](src/serving/app.py#L215-L241) — `/ingest_mlflow` reindexar métricas DVC/MLflow sem autenticação (qualquer agente externo pode disparar reindexação).
- [embedding.py:85-206](src/rag/embedding.py#L85-L206) — `ingest_documents` modifica variáveis globais (`docs`, `all_chunks`, `metadata`, `index`).
- [tools.py:56-65](src/agent/tools.py#L56-L65) — `tool_fetch_news` carrega notícias de fontes externas e as injeta no índice **sem validação de origem**, durante execução do agente.
- [embedding.py:211-214](src/rag/embedding.py#L211-L214) — ingestão automática na inicialização da API com `overwrite=True`.

**Impacto.** Manipulação de respostas (recomendações financeiras enviesadas), DoS por sobrescrita do índice (`overwrite=True`), envenenamento permanente até reinício do serviço, vetor para prompt injection indireta, contaminação do baseline de métricas indexadas.

**Mitigações recomendadas.**
1. **Autenticar `/ingest` e `/ingest_mlflow`** (token, mTLS ou JWT) — operações de escrita não podem ser públicas.
2. Validar fonte de documentos: lista de domínios permitidos, assinatura, hashes esperados.
3. Mudar default de `IngestRequest.overwrite` para `False` — `True` deve exigir flag administrativa explícita ([app.py:45](src/serving/app.py#L45)).
4. Manter índice **versionado** com possibilidade de rollback.
5. Isolar índice por tenant/sessão — atualmente é uma variável global única ([embedding.py:47-52](src/rag/embedding.py#L47-L52)).
6. Validar/sanitizar conteúdo ingerido (anti-injection patterns + detecção de PII).
7. `tool_fetch_news` deve usar fontes whitelisted, timeout, limite de tamanho e **não** ser disparada por output de LLM sem aprovação.

---

### 3.4 LLM06:2025 — Excessive Agency

**Descrição.** O agente ReAct executa ferramentas com base em saída do LLM, que pode ser manipulada por prompt injection. As ferramentas têm efeitos colaterais reais: `tool_fetch_news` baixa de fontes externas e modifica o índice global; `search_documents` pode ser usada para enumeração de conteúdo de outros usuários.

**Onde no projeto.**
- [react_agent.py:97-101](src/agent/react_agent.py#L97-L101) — `_execute_tool` chama qualquer `action` presente em `TOOL_MAP` sem aprovação humana.
- [tools.py:56-65](src/agent/tools.py#L56-L65) — `tool_fetch_news` faz I/O de rede e escreve no índice global.
- [react_agent.py:113](src/agent/react_agent.py#L113) — `max_steps` controlado por config (`config/model_config.yaml`), mas múltiplas chamadas de tool são permitidas por requisição.
- Ausência de logging estruturado das decisões de tool (apenas `trace` em memória, devolvido na resposta).

**Impacto.** Side-effects não desejados (envenenamento do índice via `fetch_news` em loop), exfiltração de dados via tool de busca, custo computacional ampliado (cada step custa uma chamada LLM remota — paga por tokens na RunPod), ausência de trilha de auditoria persistente.

**Mitigação atual.** `TOOL_MAP` fechado em [tools.py:117](src/agent/tools.py#L117) — agente não pode inventar ferramentas. Limite de `max_steps` configurável.

**Mitigações recomendadas.**
1. **Princípio do menor privilégio**: `tool_fetch_news` deveria exigir autorização explícita (role admin), não ser disparada por output de LLM.
2. Separar tools de leitura (idempotentes) de tools de escrita (com efeito colateral) e exigir confirmação para as últimas.
3. Limitar `max_steps` agressivamente em produção (ex.: 3) e instrumentar com métricas de uso (Prometheus já está no stack — expor counter por tool).
4. Logar todas as decisões de tool no Langfuse com `query` original, `action` e `action_input` para auditoria persistente.
5. Tornar a saída do LLM dado não-confiável: validar formato e *whitelist* de tools chamáveis a cada step.

---

### 3.5 API1:2023 + A05:2021 — Broken Access Control & Security Misconfiguration

**Descrição.** **Nenhum** endpoint da API exige autenticação. `/ingest`, `/ingest_mlflow`, `/query` e `/agent` são totalmente públicos. Adicionalmente, o stack de observabilidade adicionado em [docker-compose.yaml](docker-compose.yaml) introduz novas superfícies expostas com **credenciais default fracas e exposição anônima**, e o CORS continua permissivo.

**Onde no projeto.**
- [src/serving/app.py](src/serving/app.py) inteiro — sem `Depends(security)`, sem JWT, sem API key.
- [app.py:28-34](src/serving/app.py#L28-L34) — CORS com `allow_origins=["*"]` **e** `allow_credentials=True` (combinação inválida que indica configuração descuidada e habilita CSRF caso credenciais venham a ser usadas).
- [docker-compose.yaml:122-126](docker-compose.yaml#L122-L126) — Grafana com `GF_AUTH_ANONYMOUS_ENABLED=true` (role Viewer) e senha admin default `datathon2024` (em variável `GRAFANA_PASSWORD` com fallback).
- [docker-compose.yaml:146-147](docker-compose.yaml#L146-L147) — Langfuse com `NEXTAUTH_SECRET` e `SALT` defaults previsíveis (`datathon-secret-key-32chars`, `datathon-salt-key-32chars-here`).
- [docker-compose.yaml:160-162](docker-compose.yaml#L160-L162) — PostgreSQL do Langfuse com credenciais hardcoded (`langfuse:langfuse`).
- Portas expostas no host: `8000` (API), `9090` (Prometheus), `3001` (Grafana), `3000` (Langfuse), `5000` (MLflow UI).

**Impacto.** Combinado com 3.3 (poisoning), 3.4 (excessive agency) e exfiltração de dados via 3.2, eleva todas essas ameaças para criticidade máxima. Qualquer pessoa com acesso de rede ao serviço pode ler, escrever e fazer o agente executar ações. Acesso anônimo ao Grafana revela métricas operacionais; secrets fracos do Langfuse permitem forjar tokens; PostgreSQL com credenciais default permite pivoting.

**Mitigações recomendadas.**
1. Implementar **autenticação obrigatória** na API (FastAPI `Security` + `APIKeyHeader` ou OAuth2).
2. RBAC: roles distintas para `read` (`/query`), `agent` (`/agent`) e `admin` (`/ingest`, `/ingest_mlflow`).
3. Definir `allow_origins` com lista explícita de domínios confiáveis e `allow_credentials=False` enquanto não houver auth baseada em cookie.
4. Adicionar headers de segurança: `X-Content-Type-Options: nosniff`, `Strict-Transport-Security`, `Content-Security-Policy`.
5. **Grafana**: desabilitar `GF_AUTH_ANONYMOUS_ENABLED`, exigir `GRAFANA_PASSWORD` via *secret manager* (Vault, AWS Secrets Manager, Doppler) — sem default.
6. **Langfuse**: gerar `NEXTAUTH_SECRET` e `SALT` com `openssl rand -hex 32` por ambiente; rotacionar credenciais do PostgreSQL e movê-las para *secret manager*.
7. Em ambiente Docker, restringir bind das portas internas (Prometheus, Grafana, Langfuse, MLflow, PostgreSQL) a `127.0.0.1` ou rede Docker interna; expor apenas via reverse proxy autenticado (Traefik/nginx + auth).
8. Rate limiting por IP/API key (ex.: `slowapi` para FastAPI) — Prometheus já permite criar alertas de abuso.

---

## 4. Próximos passos sugeridos

| Prioridade | Ação | Responsável | Esforço |
|------------|------|-------------|---------|
| P0 | Integrar `InputGuardrail` e `OutputGuardrail` em `app.py` (todos os endpoints) | Eng. Backend | 1d |
| P0 | Autenticação obrigatória nos endpoints (`/ingest`, `/ingest_mlflow` primeiro) | Eng. Backend | 2d |
| P0 | Restringir CORS, remover credenciais default de Grafana/Langfuse, mover para secret manager | Eng. DevOps | 1d |
| P0 | PII scrubbing antes do envio ao endpoint vLLM remoto (RunPod) | Eng. ML | 1d |
| P1 | Mudar default de `IngestRequest.overwrite` para `False`; adicionar limites de tamanho | Eng. Backend | 0.5d |
| P1 | Rate limiting na API (`slowapi`) + alertas no Prometheus | Eng. DevOps | 1d |
| P1 | Restringir bind de portas internas (Prometheus, Grafana, MLflow, PostgreSQL) a rede Docker | Eng. DevOps | 0.5d |
| P2 | Particionamento do índice FAISS por tenant/sessão | Eng. ML | 3d |
| P2 | Logging estruturado de tool calls do agente no Langfuse | Eng. ML | 1d |
| P2 | Detecção de PII brasileiro adicional (CNPJ, RG, dados bancários) | Eng. Segurança | 1d |

---

## 5. Referências

- [OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/llm-top-10/)
- [OWASP API Security Top 10 (2023)](https://owasp.org/API-Security/editions/2023/en/0x11-t10/)
- [OWASP Top 10 (2021)](https://owasp.org/Top10/)
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [Microsoft Threat Modeling for AI/ML systems](https://learn.microsoft.com/en-us/security/ai-red-team/)
