# OWASP — Análise de Ameaças do Projeto

Documento de mapeamento de ameaças de segurança aplicáveis ao **Datathon MLET — Grupo 05**, um sistema de **RAG + Agente ReAct** servido via **FastAPI**, com geração de texto delegada a um **LLM** (BentoML/vLLM ou fallback Hugging Face) e índice vetorial **FAISS**.

A análise utiliza como referência principal a **[OWASP Top 10 for LLM Applications (2025)](https://owasp.org/www-project-top-10-for-large-language-model-applications/)**, complementada com itens da **OWASP API Security Top 10** quando relevantes para a superfície HTTP exposta pela aplicação.

---

## 1. Contexto do sistema sob análise

| Componente | Localização | Exposição |
|------------|-------------|-----------|
| API HTTP | [src/serving/app.py](src/serving/app.py) | Endpoints públicos `/ingest`, `/query`, `/agent` |
| Agente ReAct | [src/agent/react_agent.py](src/agent/react_agent.py) | Executa ferramentas (`search_documents`, `fetch_news`, `summarize_context`) |
| Pipeline RAG | [src/rag/embedding.py](src/rag/embedding.py), [src/rag/generator.py](src/rag/generator.py) | Embeddings (SentenceTransformers) + FAISS + LLM remoto/local |
| Guardrails | [src/security/guardrails.py](src/security/guardrails.py) | Detecção de prompt injection (regex) + PII output (Presidio) |
| Geração LLM | BentoML/vLLM remoto (`RAG_GENERATOR_URL`) ou Hugging Face local (`RAG_MODEL`) | Configurável via variável de ambiente |

Superfície de ataque principal: **entradas de texto não confiáveis enviadas pelos endpoints HTTP**, que alimentam o LLM, o índice vetorial e as chamadas de ferramentas do agente.

---

## 2. Resumo executivo das ameaças mapeadas

| # | ID OWASP | Ameaça | Severidade | Status atual |
|---|----------|--------|------------|--------------|
| 1 | LLM01:2025 | Prompt Injection (direta e indireta via RAG) | **Alta** | Mitigação parcial (regex em [guardrails.py:15-28](src/security/guardrails.py#L15-L28)) |
| 2 | LLM02:2025 | Sensitive Information Disclosure (PII / segredos) | **Alta** | Mitigação parcial (Presidio em output) |
| 3 | LLM04:2025 | Data and Model Poisoning via `/ingest` | **Alta** | **Sem mitigação** |
| 4 | LLM05:2025 | Improper Output Handling (XSS via `api_test.html`) | **Média** | **Sem mitigação** |
| 5 | LLM06:2025 | Excessive Agency do agente ReAct | **Média** | Mitigação parcial (TOOL_MAP fechado) |
| 6 | LLM08:2025 | Vector & Embedding Weaknesses (FAISS sem isolamento) | **Média** | **Sem mitigação** |
| 7 | LLM09:2025 | Misinformation (recomendações financeiras alucinadas) | **Alta** | **Sem mitigação** |
| 8 | LLM10:2025 | Unbounded Consumption (DoS / custo) | **Média** | Mitigação parcial (`max_new_tokens`) |
| 9 | API1:2023 / A01:2021 | Broken Access Control (endpoints sem autenticação) | **Crítica** | **Sem mitigação** |
| 10 | A05:2021 | Security Misconfiguration (CORS `*` + credentials) | **Alta** | **Sem mitigação** |

---

## 3. Ameaças detalhadas

### 3.1 LLM01:2025 — Prompt Injection

**Descrição.** Um atacante injeta instruções que sobrescrevem o *system prompt* do agente, forçando o LLM a ignorar restrições, vazar o prompt do agente, executar ferramentas indevidas ou produzir respostas maliciosas. Pode ser **direta** (no campo `query` de `/agent` ou `/query`) ou **indireta** (via documentos enviados em `/ingest` ou notícias coletadas por `tool_fetch_news`).

**Onde no projeto.**
- Entrada direta: [app.py:50-58](src/serving/app.py#L50-L58) (`/query`) e [app.py:66-69](src/serving/app.py#L66-L69) (`/agent`).
- Entrada indireta: [embedding.py:81-130](src/rag/embedding.py#L81-L130) (qualquer texto ingerido vira contexto do LLM em consultas futuras).
- Prompt do agente embutido no código: [react_agent.py:59-87](src/agent/react_agent.py#L59-L87).

**Impacto.** Vazamento do system prompt, execução não autorizada de tools (`fetch_news` para forçar coleta externa, `search_documents` para enumeração), exfiltração de contexto de outros usuários (já que o índice FAISS é **global e compartilhado**), respostas tendenciosas/maliciosas.

**Mitigação atual.** [guardrails.py:15-53](src/security/guardrails.py#L15-L53) bloqueia ~12 padrões regex em inglês ("ignore previous instructions", "DAN mode", etc.). **Limitações:** (a) o `InputGuardrail` **não é chamado** no `app.py` — está implementado mas não integrado; (b) padrões em português não cobertos ("ignore as instruções acima"); (c) não cobre injeção indireta via `/ingest`.

**Mitigações recomendadas.**
1. **Integrar `InputGuardrail.validate()` em `/query` e `/agent`** antes de chegar ao LLM.
2. Aplicar guardrail também sobre documentos em `/ingest` (injeção indireta).
3. Usar separadores estruturados/delimitadores no prompt do agente e/ou *spotlighting* para distinguir instrução de dados.
4. Avaliar classificadores ML para detecção de injection (ex.: `protectai/deberta-v3-base-prompt-injection-v2`).
5. Não confiar na saída do LLM para decisões privilegiadas — sempre validar antes de executar tools.

---

### 3.2 LLM02:2025 — Sensitive Information Disclosure

**Descrição.** O LLM pode reproduzir PII presente nos documentos ingeridos, vazar tokens/segredos do ambiente, ou regurgitar trechos de seu corpus de treino contendo dados sensíveis.

**Onde no projeto.**
- Variáveis sensíveis: `HF_TOKEN`, `HUGGINGFACEHUB_API_TOKEN` em [generator.py:18-22](src/rag/generator.py#L18-L22) e [embedding.py:24-28](src/rag/embedding.py#L24-L28). Se acidentalmente impressas em log de erro ou colocadas no contexto, podem ser ecoadas.
- Documentos ingeridos via `/ingest` podem conter CPF, e-mail, telefone, nome de pessoas — e são devolvidos como `context` em `/query` ([app.py:55-58](src/serving/app.py#L55-L58)).

**Impacto.** Vazamento de PII (violação **LGPD**), exposição de credenciais HF (uso indevido de cota), exposição de dados confidenciais corporativos.

**Mitigação atual.** [guardrails.py:56-92](src/security/guardrails.py#L56-L92) — `OutputGuardrail` com Presidio detectando `PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `BR_CPF`. **Limitações:** (a) não está integrado em `app.py`; (b) `/query` retorna o `context` bruto sem sanitização ([app.py:58](src/serving/app.py#L58)); (c) cobertura de PII brasileiro restrita (faltam RG, CNPJ, dados bancários).

**Mitigações recomendadas.**
1. Aplicar `OutputGuardrail.sanitize()` em **toda** resposta de `/query`, `/agent` — incluindo o campo `context`.
2. Adicionar entidades adicionais ao Presidio: `BR_CNPJ`, `CREDIT_CARD`, `IBAN_CODE`, `IP_ADDRESS`.
3. Sanitizar PII também na **entrada** (em `/ingest`) — não armazenar PII no índice.
4. Garantir que `HF_TOKEN` **nunca** entre no prompt; revisar logs (`print(f"...{exc}...")`) para evitar vazamento.
5. Política explícita no system prompt: "nunca repita números, e-mails ou nomes próprios extraídos do contexto".

---

### 3.3 LLM04:2025 — Data and Model Poisoning

**Descrição.** O endpoint `/ingest` aceita **qualquer** lista de documentos sem autenticação e os adiciona ao índice FAISS global, contaminando os resultados de **todos** os usuários subsequentes. Um atacante pode plantar documentos com instruções maliciosas (injeção indireta), desinformação financeira ou dados falsos que serão recuperados como "contexto autoritativo".

**Onde no projeto.**
- [app.py:32-47](src/serving/app.py#L32-L47) — `/ingest` aceita `IngestRequest` sem autenticação, com `overwrite=True` por padrão (permitindo substituir a base inteira).
- [embedding.py:81-130](src/rag/embedding.py#L81-L130) — `ingest_documents` modifica variáveis globais (`docs`, `all_chunks`, `metadata`, `index`).
- [tools.py:55-64](src/agent/tools.py#L55-L64) — `tool_fetch_news` carrega notícias de fontes externas e as injeta no índice **sem validação de origem**.

**Impacto.** Manipulação de respostas (recomendações financeiras enviesadas), DoS por sobrescrita do índice (`overwrite=True`), envenenamento permanente até reinício do serviço, vetor para prompt injection indireta.

**Mitigações recomendadas.**
1. **Autenticar `/ingest`** (token, mTLS ou JWT) — operação de escrita não pode ser pública.
2. Validar fonte de documentos: lista de domínios permitidos, assinatura, hashes esperados.
3. Manter índice **versionado** com possibilidade de rollback; nunca permitir `overwrite=True` sem confirmação.
4. Isolar índice por tenant/sessão — atualmente o índice é uma variável global única.
5. Validar/sanitizar conteúdo ingerido (anti-injection patterns + detecção de PII).
6. `tool_fetch_news` deve usar fontes whitelisted e timeout/limite de tamanho.

---

### 3.4 LLM05:2025 — Improper Output Handling

**Descrição.** A saída do LLM é incorporada em uma página HTML (`api_test.html`) sem escape, abrindo XSS se o modelo gerar `<script>` ou markdown malicioso (cenário possível via prompt injection ou `_generate_simulated_answer`).

**Onde no projeto.**
- [api_test.html](api_test.html) consome `/query` e `/agent`; se inserir o campo `answer` ou `context` via `innerHTML`, há XSS.
- [generator.py:101-134](src/rag/generator.py#L101-L134) — não há sanitização da string retornada pelo LLM nem do contexto.

**Impacto.** XSS no painel de testes, *session hijacking* se houver cookies, comandos disfarçados em respostas que sejam encaminhadas a downstream (e-mail, Slack, etc.).

**Mitigações recomendadas.**
1. Em `api_test.html`, usar `textContent` (não `innerHTML`) para renderizar respostas.
2. Aplicar política CSP `default-src 'self'; script-src 'self'`.
3. No backend, escapar/normalizar saída antes de retornar (remover tags HTML, controlar markdown).
4. Tratar saída do LLM como **dado não confiável**, mesmo dentro do próprio backend (não usar como base para chamadas a outras APIs sem validação).

---

### 3.5 LLM06:2025 — Excessive Agency

**Descrição.** O agente ReAct executa ferramentas com base em saída do LLM, que pode ser manipulada por prompt injection. As ferramentas têm efeitos colaterais reais: `tool_fetch_news` baixa de fontes externas e modifica o índice; `search_documents` pode ser usada para enumeração.

**Onde no projeto.**
- [react_agent.py:90-94](src/agent/react_agent.py#L90-L94) — `_execute_tool` chama qualquer `action` presente em `TOOL_MAP` sem aprovação humana.
- [tools.py:55-64](src/agent/tools.py#L55-L64) — `tool_fetch_news` faz I/O de rede e escreve no índice global.
- [react_agent.py:98](src/agent/react_agent.py#L98) — `max_steps` controlado por config, mas até 10 chamadas de tool são permitidas por requisição.

**Impacto.** Side-effects não desejados (envenenamento do índice via `fetch_news` em loop), exfiltração de dados via tool de busca, custo computacional ampliado (cada step custa uma chamada LLM).

**Mitigação atual.** `TOOL_MAP` fechado em [tools.py:116](src/agent/tools.py#L116) — agente não pode inventar ferramentas. Limite de `max_steps`.

**Mitigações recomendadas.**
1. **Princípio do menor privilégio**: `tool_fetch_news` deveria exigir autorização explícita, não ser disparada por output de LLM.
2. Separar tools de leitura (idempotentes) de tools de escrita (com efeito colateral) e exigir confirmação para as últimas.
3. Limitar `max_steps` agressivamente em produção (ex.: 3) e instrumentar com métricas de uso.
4. Logar todas as decisões de tool com `query` original, `action` e `action_input` para auditoria.

---

### 3.6 LLM08:2025 — Vector and Embedding Weaknesses

**Descrição.** O índice FAISS é **global, sem isolamento por usuário/tenant** e mantido em memória do processo. Qualquer documento ingerido por um usuário fica disponível para qualquer outro via `/query` ou `search_documents`. Adicionalmente, embeddings podem ser invertidos para reconstrução parcial do texto original (*embedding inversion attacks*).

**Onde no projeto.**
- [embedding.py:46-49](src/rag/embedding.py#L46-L49) — `docs`, `all_chunks`, `metadata`, `index` são variáveis globais do módulo.
- [embedding.py:75-76](src/rag/embedding.py#L75-L76) — `IndexFlatL2` sem criptografia nem ACL.

**Impacto.** Vazamento *cross-tenant* de informações ingeridas, possibilidade de reconstrução de texto a partir dos embeddings se eles forem expostos, ausência de auditoria.

**Mitigações recomendadas.**
1. Particionar o índice por chave de tenant/sessão (FAISS suporta múltiplos índices) e filtrar por escopo no `retrieve`.
2. Persistir índice em store apropriado com controle de acesso (ex.: Qdrant/Weaviate com auth).
3. Aplicar embeddings privacy-preserving para dados sensíveis ou criptografar campos de metadata.
4. Auditar todos os documentos atualmente no índice — `print` em [embedding.py:125](src/rag/embedding.py#L125) confirma ingestão automática de notícias de fontes externas sem revisão.

---

### 3.7 LLM09:2025 — Misinformation

**Descrição.** O sistema **explicitamente** entrega recomendações financeiras: ações sugeridas, percentuais de alocação, "compre Bitcoin" — geradas por LLM ou por respostas simuladas hardcoded em [generator.py:137-171](src/rag/generator.py#L137-L171). Em caso de uso indevido por usuário leigo, há risco de prejuízo financeiro real e responsabilidade legal.

**Onde no projeto.**
- [generator.py:148-161](src/rag/generator.py#L148-L161) — string literal: "recomendo priorizar investimentos em títulos de renda fixa como CDB, Tesouro Direto e debêntures".
- [react_agent.py:60-62](src/agent/react_agent.py#L60-L62) — system prompt: "agente ReAct especializado em finanças e mercado financeiro".

**Impacto.** Decisões financeiras tomadas com base em alucinação ou em respostas simuladas estáticas; possíveis violações de regulamentação (CVM no Brasil — recomendação de investimentos exige autorização).

**Mitigações recomendadas.**
1. **Disclaimer obrigatório** em toda resposta: "Este sistema não substitui um assessor financeiro autorizado pela CVM".
2. Remover ou marcar claramente como *simulada/educacional* a função `_generate_simulated_answer`.
3. Aterrar respostas em fontes verificáveis e exibir as referências (já há `metadata` disponível em [retriever.py](src/rag/retriever.py)).
4. Avaliação humana periódica das saídas (RLHF / red team financeiro).
5. Detector de hallucination (ex.: comparar entidades da resposta com entidades do contexto recuperado).

---

### 3.8 LLM10:2025 — Unbounded Consumption

**Descrição.** Endpoints aceitam qualquer payload sem rate limiting, sem limite de tamanho de documentos em `/ingest` e sem cota de tokens por requisição. Um atacante pode esgotar quota de API HF/vLLM, encher memória com documentos enormes ou consumir CPU em embeddings.

**Onde no projeto.**
- [app.py](src/serving/app.py) — nenhum middleware de rate limit.
- [app.py:32-47](src/serving/app.py#L32-L47) — `/ingest` sem limite de quantidade ou tamanho de documentos.
- [react_agent.py:106-156](src/agent/react_agent.py#L106-L156) — cada chamada a `/agent` faz `max_steps` chamadas LLM.

**Mitigação atual.** `_INFERENCE_MAX_NEW_TOKENS` controla tokens gerados por chamada; `InputGuardrail` (não integrado) limita input a 4096 chars.

**Mitigações recomendadas.**
1. Rate limit por IP/API key (ex.: `slowapi` para FastAPI).
2. Limitar `IngestRequest.docs` a N documentos e M chars/doc (validação Pydantic).
3. Timeout agressivo em `_call_bento_generator` (atualmente 15s — ok, mas precisa de circuit breaker).
4. Cota mensal de tokens por API key.
5. Monitoramento de custo (MLflow já é usado para experimentos — estender para inference).

---

### 3.9 API1:2023 / A01:2021 — Broken Access Control

**Descrição.** **Nenhum** endpoint da API exige autenticação. `/ingest` (escrita), `/query` (leitura) e `/agent` (execução de tools) são totalmente públicos.

**Onde no projeto.** [src/serving/app.py](src/serving/app.py) inteiro — sem `Depends(security)`, sem JWT, sem API key.

**Impacto.** Combinado com 3.3 (poisoning), 3.5 (excessive agency) e 3.8 (unbounded consumption), eleva todas essas ameaças para criticidade máxima. Qualquer pessoa com acesso de rede ao serviço pode ler, escrever e fazer o agente executar ações.

**Mitigações recomendadas.**
1. Implementar **autenticação obrigatória** (FastAPI `Security` + `APIKeyHeader` ou OAuth2).
2. RBAC: roles distintas para `read` (`/query`), `agent` (`/agent`) e `admin` (`/ingest`).
3. Em ambiente Docker, restringir bind de portas a `127.0.0.1` quando exposto via reverse proxy.

---

### 3.10 A05:2021 — Security Misconfiguration (CORS permissivo)

**Descrição.** Em [app.py:12-18](src/serving/app.py#L12-L18), o CORS está configurado com `allow_origins=["*"]` **e** `allow_credentials=True` — combinação que browsers já bloqueiam, mas que indica configuração descuidada e habilita CSRF caso credenciais venham a ser usadas.

**Onde no projeto.** [src/serving/app.py:12-18](src/serving/app.py#L12-L18).

**Impacto.** Permite que qualquer página externa invoque a API a partir do browser do usuário; quando autenticação for adicionada, abre vetor de CSRF.

**Mitigações recomendadas.**
1. Definir `allow_origins` com lista explícita de domínios confiáveis.
2. Manter `allow_credentials=False` enquanto não houver auth baseada em cookie.
3. Restringir `allow_methods` ao mínimo necessário (já está restrito, ok).
4. Adicionar headers de segurança: `X-Content-Type-Options: nosniff`, `Strict-Transport-Security`, `Content-Security-Policy`.

---

## 4. Próximos passos sugeridos

| Prioridade | Ação | Responsável | Esforço |
|------------|------|-------------|---------|
| P0 | Integrar `InputGuardrail` e `OutputGuardrail` em `app.py` | Eng. Backend | 1d |
| P0 | Autenticação obrigatória nos endpoints (especialmente `/ingest`) | Eng. Backend | 2d |
| P0 | Restringir CORS e adicionar security headers | Eng. Backend | 0.5d |
| P1 | Rate limiting + limites de tamanho em `/ingest` | Eng. Backend | 1d |
| P1 | Disclaimer financeiro + remover respostas simuladas hardcoded | Produto | 0.5d |
| P2 | Particionamento do índice FAISS por tenant | Eng. ML | 3d |
| P2 | Auditoria/logging estruturado de tool calls do agente | Eng. ML | 1d |
| P2 | Detecção de PII brasileiro adicional (CNPJ, RG, dados bancários) | Eng. Segurança | 1d |

---

## 5. Referências

- [OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/llm-top-10/)
- [OWASP API Security Top 10 (2023)](https://owasp.org/API-Security/editions/2023/en/0x11-t10/)
- [OWASP Top 10 (2021)](https://owasp.org/Top10/)
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [Microsoft Threat Modeling for AI/ML systems](https://learn.microsoft.com/en-us/security/ai-red-team/)
