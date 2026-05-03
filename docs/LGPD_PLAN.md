# Plano de Adequação à LGPD — Datathon MLET (Grupo 05)

Plano de conformidade à **Lei Geral de Proteção de Dados Pessoais (Lei nº 13.709/2018)** aplicado ao sistema atual: **RAG + Agente ReAct + FastAPI + FAISS + LLM remoto (vLLM/RunPod)** ([src/serving/app.py](src/serving/app.py)) — com *stack* de observabilidade **Prometheus + Grafana + Langfuse** ([docker-compose.yaml](docker-compose.yaml)).

Este documento traduz os requisitos da LGPD em ações **executáveis sobre o código real** do projeto, em complemento a [docs/OWASP.md](docs/OWASP.md) e [docs/RED_TEAM_REPORT.md](docs/RED_TEAM_REPORT.md). Cada item da LGPD é mapeado para o componente do sistema afetado e para uma ação concreta com prioridade.

> **Aviso.** Documento técnico-operacional. Não substitui parecer jurídico. Para interpretação de bases legais e revisão de DPIA/RIPD, consultar profissional habilitado.

---

## Sumário

1. [Contexto e escopo](#1-contexto-e-escopo)
2. [Mapeamento de dados pessoais (Registro de Operações de Tratamento)](#2-mapeamento-de-dados-pessoais-registro-de-operações-de-tratamento)
3. [Bases legais aplicáveis (Art. 7º e Art. 11)](#3-bases-legais-aplicáveis-art-7º-e-art-11)
4. [Princípios da LGPD aplicados ao projeto (Art. 6º)](#4-princípios-da-lgpd-aplicados-ao-projeto-art-6º)
5. [Direitos dos titulares (Art. 18) — como atender](#5-direitos-dos-titulares-art-18--como-atender)
6. [Medidas de segurança técnicas e organizacionais (Art. 46–49)](#6-medidas-de-segurança-técnicas-e-organizacionais-art-4649)
7. [Relatório de Impacto à Proteção de Dados — RIPD (Art. 38)](#7-relatório-de-impacto-à-proteção-de-dados--ripd-art-38)
8. [Encarregado (DPO) — Art. 41](#8-encarregado-dpo--art-41)
9. [Transferências internacionais (Art. 33)](#9-transferências-internacionais-art-33)
10. [Plano de retenção e eliminação (Art. 15–16)](#10-plano-de-retenção-e-eliminação-art-1516)
11. [Gestão de incidentes (Art. 48)](#11-gestão-de-incidentes-art-48)
12. [Roadmap de adequação](#12-roadmap-de-adequação)
13. [Checklist de conformidade](#13-checklist-de-conformidade)

---

## 1. Contexto e escopo

**Sistema.** Aplicação de IA generativa que recebe documentos textuais via `/ingest`, métricas de modelos via `/ingest_mlflow`, indexa em vetor FAISS e responde a perguntas em linguagem natural via `/query` e `/agent`, com possíveis recomendações financeiras (ver [src/rag/generator.py:569+](src/rag/generator.py#L569)). A geração de texto é delegada a um **endpoint vLLM remoto** quando `VLLM_BASE_URL` está configurado ([src/rag/generator.py:12](src/rag/generator.py#L12), [src/rag/generator.py:234](src/rag/generator.py#L234)).

**Por que a LGPD se aplica.**
- Documentos ingeridos podem conter **dados pessoais** (nomes, CPF, e-mail, telefone) — conforme já demonstrado pelo `OutputGuardrail` declarar essas entidades em [src/security/guardrails.py:85-92](src/security/guardrails.py#L85-L92).
- A query do próprio usuário é **dado pessoal** (associada a IP/sessão) e pode revelar interesses financeiros (categoria sensível para perfilamento).
- Logs (`print(...)` em [src/rag/embedding.py](src/rag/embedding.py), [src/rag/generator.py](src/rag/generator.py)) e MLflow ([src/rag/embedding.py:180-193](src/rag/embedding.py#L180-L193)) persistem evidências de tratamento.
- O **Langfuse** ([docker-compose.yaml:138-154](docker-compose.yaml#L138-L154)) — quando integrado — armazena prompts e contextos completos em PostgreSQL próprio, criando novo locus de retenção de dados pessoais.

**Papéis.**
- **Controlador**: o time/empresa que opera a aplicação (define finalidade e meios — Art. 5º, VI).
- **Operadores**: provedores que executam tratamento em nome do controlador:
  - **Hugging Face** (download/inference de modelos — `RAG_MODEL`, `HF_TOKEN`).
  - **RunPod / provedor de vLLM remoto** — recebe `prompt + context` completos a cada chamada de geração (`VLLM_BASE_URL`).
  - **BentoML** (camada de serviço local — `RAG_GENERATOR_URL`, opcional).
  - **Langfuse** (observabilidade de LLM com armazenamento próprio de prompts/respostas).
  - **Grafana / Prometheus** (telemetria operacional — pode incidentalmente capturar dados pessoais em labels de métricas).

---

## 2. Mapeamento de dados pessoais (Registro de Operações de Tratamento)

Requerido pelo **Art. 37 da LGPD**.

| # | Dado pessoal | Onde é tratado | Categoria | Origem | Finalidade |
|---|--------------|----------------|-----------|--------|------------|
| 1 | Texto de query do usuário | [app.py:244-262](src/serving/app.py#L244-L262), [app.py:270-275](src/serving/app.py#L270-L275) | Comum (pode conter sensível por inferência) | Coleta direta | Geração de resposta |
| 2 | Conteúdo de documentos ingeridos (pode incluir CPF, nome, e-mail, telefone, endereço) | [app.py:187-241](src/serving/app.py#L187-L241), [embedding.py:85-206](src/rag/embedding.py#L85-L206) | Comum + potencialmente sensível | Upload via API / `tool_fetch_news` / `/ingest_mlflow` | Construção da base de conhecimento |
| 3 | Embeddings vetoriais derivados | FAISS in-memory ([embedding.py:47-52](src/rag/embedding.py#L47-L52)) | Pseudonimizado, mas **reversível** (embedding inversion) | Derivado de #2 | Busca semântica |
| 4 | Metadados de runs MLflow | [embedding.py:180-193](src/rag/embedding.py#L180-L193) | Comum | Sistema | Rastreabilidade do experimento |
| 5 | Prompts/respostas armazenados no Langfuse | PostgreSQL do Langfuse ([docker-compose.yaml:156-172](docker-compose.yaml#L156-L172)) | Comum + potencialmente sensível | Sistema | Observabilidade do LLM |
| 6 | Métricas operacionais (Prometheus) | [docker-compose.yaml:96-113](docker-compose.yaml#L96-L113) | Comum (incidental) | Sistema | Monitoramento |
| 7 | Logs de aplicação | `print(...)` espalhados, stack traces | Comum | Sistema | Debug |
| 8 | IP de origem da requisição | Cabeçalhos HTTP (FastAPI) | Comum | Coleta automática | Identificação de sessão |
| 9 | Tokens Hugging Face (`HF_TOKEN`) e vLLM (`VLLM_API_KEY`) | Variáveis de ambiente | **Não pessoal**, mas credencial sensível | Configuração | Acesso à API HF / endpoint vLLM remoto |
| 10 | Prompt completo enviado à RunPod | Tráfego HTTP para `VLLM_BASE_URL` ([generator.py:177-179](src/rag/generator.py#L177-L179)) | Pode conter dado pessoal de #2 | Derivado | Geração de texto |

**Lacunas identificadas.**
- **Não há** mecanismo para distinguir dados pessoais dos não pessoais no índice FAISS.
- **Não há** registro temporal de quando cada documento foi ingerido (campo `fetched_at` existe na metadata em [embedding.py:64-72](src/rag/embedding.py#L64-L72), mas só é preenchido quando vem do `data_loader.load_news`).
- **Não há** vínculo entre `doc_id` e a identidade do remetente — auditoria reversa impossível.
- **Não há** sanitização de PII antes do envio do prompt ao vLLM remoto — todo dado pessoal sai da fronteira da aplicação.

**Ação P0.** Criar documento operacional `data/registro_tratamento.md` com este registro, atualizado a cada nova fonte de dados.

---

## 3. Bases legais aplicáveis (Art. 7º e Art. 11)

| Tratamento | Base legal sugerida | Justificativa |
|------------|---------------------|---------------|
| Query do usuário → resposta | **Execução de contrato** (Art. 7º, V) ou **legítimo interesse** (Art. 7º, IX) | Necessário para entregar o serviço solicitado |
| Documentos ingeridos contendo PII de terceiros (não o usuário) | **Consentimento** (Art. 7º, I) do titular original **OU** legítimo interesse com teste de balanceamento documentado | Dado pessoal de **terceiro** — consentimento do *uploader* não basta |
| Logs e métricas para segurança | **Cumprimento de obrigação legal** (Art. 7º, II) + legítimo interesse | Detecção de fraude/incidente — Art. 48 |
| Modelo treinado com dados pessoais | **Base mais restritiva aplicável a cada dado** | Treinamento ≠ inferência — bases podem divergir |
| Dados sensíveis (saúde, opinião política, biometria) | **Consentimento específico e destacado** (Art. 11, I) | **Não há previsão atual de tratar dado sensível** — proibir explicitamente em `/ingest` |

**Ação P0.** Documentar em [src/serving/app.py](src/serving/app.py) (docstring do endpoint) a base legal de cada operação. **Recusar dados sensíveis** no `/ingest` com validação explícita (atualmente nada bloqueia ingestão de prontuário médico, p.ex.).

---

## 4. Princípios da LGPD aplicados ao projeto (Art. 6º)

| Princípio | Status atual | Ação requerida |
|-----------|--------------|----------------|
| **Finalidade** (I) | ❌ Indefinida — `/ingest` aceita qualquer doc | Definir finalidade explícita ("análise de mercado financeiro") e validar coerência |
| **Adequação** (II) | ❌ | Documentar compatibilidade entre dado coletado e finalidade |
| **Necessidade** (III) | ❌ Coleta excessiva (texto bruto inteiro) | Pré-processar/redigir PII na ingestão antes de indexar |
| **Livre acesso** (IV) | ❌ Sem endpoint para titular consultar | Implementar `/lgpd/access` (ver §5) |
| **Qualidade dos dados** (V) | ⚠️ Sem validação de origem | Whitelist de fontes em `tool_fetch_news` |
| **Transparência** (VI) | ❌ Sem aviso de privacidade | Publicar Política de Privacidade e expor em `GET /lgpd/policy` |
| **Segurança** (VII) | ❌ Sem auth, CORS aberto, defaults fracos no Grafana/Langfuse | Ver [docs/OWASP.md](docs/OWASP.md) §3.5 |
| **Prevenção** (VIII) | ⚠️ Guardrails existem mas não integrados | Integrar [src/security/guardrails.py](src/security/guardrails.py) em `app.py` |
| **Não discriminação** (IX) | ⚠️ Recomendações financeiras sem auditoria de viés | Avaliação periódica de outputs por subgrupos |
| **Responsabilização e prestação de contas** (X) | ❌ Sem trilha de auditoria | Logging estruturado por requisição com `request_id` |

---

## 5. Direitos dos titulares (Art. 18) — como atender

A LGPD garante 9 direitos. Hoje o sistema **não atende a nenhum** porque (a) não há autenticação, (b) não há vínculo entre `doc_id` e titular, (c) o índice FAISS não suporta deleção seletiva eficiente.

| Direito (Art. 18) | Endpoint a criar | Implementação |
|-------------------|------------------|---------------|
| I — Confirmação de tratamento | `GET /lgpd/me` | Após auth, retornar lista de `doc_id` associados ao requerente |
| II — Acesso aos dados | `GET /lgpd/me/data` | Devolver todos os textos originais e contextos onde aparece |
| III — Correção | `PUT /lgpd/me/data/{doc_id}` | Atualiza documento e re-indexa |
| IV — Anonimização / bloqueio / eliminação de dados desnecessários | `POST /lgpd/me/anonymize` | Aplica Presidio + reembedding |
| V — Portabilidade | `GET /lgpd/me/export` | Export JSON completo |
| VI — **Eliminação** dos dados tratados com consentimento | `DELETE /lgpd/me/data/{doc_id}` | **Bloqueador técnico atual**: FAISS `IndexFlatL2` não suporta `remove`; é preciso rebuildar |
| VII — Compartilhamento com terceiros | `GET /lgpd/me/sharing` | Listar operadores (HF, RunPod) que receberam dados |
| VIII — Possibilidade de não consentir | UI antes de `/ingest` | Termo de consentimento explícito e granular |
| IX — Revogação do consentimento | `POST /lgpd/me/revoke` | Aciona fluxo do direito VI |

**Bloqueadores técnicos a resolver primeiro.**
1. **Identidade do titular**: hoje não existe. Adicionar autenticação (P0 — ver [docs/OWASP.md](docs/OWASP.md) §3.5).
2. **Vínculo titular → documento**: estender metadata em [embedding.py:64-72](src/rag/embedding.py#L64-L72) com campo `subject_id` ou `tenant_id`.
3. **Deleção em FAISS**: migrar para `IndexIDMap` + `remove_ids()` ou para vetor store com suporte nativo a deleção (Qdrant/Weaviate).
4. **Reversão do embedding**: ao deletar texto, deletar **também** chunks, embeddings, logs MLflow e traces no Langfuse correspondentes — não apenas o documento original.

**Prazo de atendimento.** Art. 19, §1º — **15 dias** para confirmação/acesso. Art. 18, §3º — em geral imediato/no prazo razoável para os demais. Implementar SLA monitorado.

---

## 6. Medidas de segurança técnicas e organizacionais (Art. 46–49)

Mapeamento das mitigações já listadas em [docs/OWASP.md](docs/OWASP.md) §4 e [docs/RED_TEAM_REPORT.md](docs/RED_TEAM_REPORT.md), categorizadas conforme exigência da LGPD.

### Técnicas
| Controle | Status | Ação |
|----------|--------|------|
| Autenticação e autorização | ❌ Ausente | API key + RBAC (`read`/`agent`/`admin`) — P0 |
| Criptografia em trânsito | ⚠️ Depende do deploy | TLS obrigatório no reverse proxy |
| Criptografia em repouso | ❌ FAISS in-memory; MLflow local | Persistir índice em store cifrado; cifrar volumes Docker |
| Segregação por tenant | ❌ Índice global | Particionar FAISS por `tenant_id` — P1 |
| Anonimização/pseudonimização na ingestão | ❌ Não aplicada | `OutputGuardrail.sanitize()` também na entrada — P0 |
| Detecção de PII nos outputs | ⚠️ Implementado, **não chamado** | Integrar `OutputGuardrail` em `/query` e `/agent` — P0 |
| Logging com redaction de PII | ❌ `print(f"{exc}")` pode vazar | Substituir por logger estruturado com filtro |
| Backup e recuperação | ❌ Não definido | Política de backup do índice + MLflow runs |
| Monitoramento de acesso | ❌ Sem audit trail | `request_id` + log estruturado por endpoint — P1 |
| Atualização de dependências | ⚠️ Poetry presente | `safety`/`pip-audit` em CI — P1 |

### Organizacionais
| Controle | Status | Ação |
|----------|--------|------|
| Política de Privacidade | ❌ Inexistente | Redigir e publicar em `GET /lgpd/policy` — P0 |
| Termo de Consentimento | ❌ | Modal antes de `/ingest` — P0 |
| Treinamento da equipe | ❌ | Capacitação anual (LGPD + uso responsável de IA) |
| Contratos com operadores (DPA) | ⚠️ Termos HF/RunPod/Langfuse aceitos implicitamente | Revisar DPA da Hugging Face, RunPod e Langfuse Cloud — Art. 39 |
| Procedimento de incidente | ❌ | Ver §11 |
| Designação do Encarregado | ❌ | Ver §8 |

---

## 7. Relatório de Impacto à Proteção de Dados — RIPD (Art. 38)

A ANPD pode exigir RIPD quando o tratamento envolver risco elevado. Este projeto **se enquadra** porque:
- Trata dados pessoais em **larga escala** potencial (sem limite de volume em `/ingest`).
- Usa **decisão automatizada** que produz efeitos relevantes (recomendação financeira — Art. 20).
- Combina dados de múltiplas fontes com **inferência semântica** (perfilamento).

### Estrutura mínima do RIPD a elaborar

1. **Descrição do tratamento** (cobre §2 deste documento).
2. **Necessidade e proporcionalidade** — por que RAG global? Pode ser substituído por contexto efêmero por sessão?
3. **Riscos identificados**:
   - Reidentificação via embedding inversion (FAISS global, sem isolamento).
   - Vazamento cross-tenant e exfiltração para o vLLM remoto (RT-03 em [docs/RED_TEAM_REPORT.md](docs/RED_TEAM_REPORT.md)).
   - Discriminação em recomendações financeiras (Art. 20).
   - Uso indevido por prompt injection indireta (RT-02).
   - Exposição de prompts/contextos via Langfuse com credenciais default (RT-05).
4. **Medidas mitigadoras** — vincular ao Roadmap §12.
5. **Decisão automatizada** — Art. 20 garante direito à revisão. Implementar `POST /lgpd/review` que abre ticket para análise humana.
6. **Aprovação do Encarregado**.

**Ação P0.** Iniciar RIPD com template ANPD ([anpd.gov.br/ripd](https://www.gov.br/anpd/pt-br)).

---

## 8. Encarregado (DPO) — Art. 41

**Obrigatório** para controladores (Art. 41, *caput*). Atribuições mínimas (Art. 41, §2º):
- Aceitar reclamações dos titulares e prestar esclarecimentos.
- Receber comunicações da ANPD.
- Orientar a equipe.
- Executar o plano de adequação.

**Ação P0.**
1. Designar Encarregado nominalmente (pode ser PJ ou PF).
2. Publicar e-mail de contato em `GET /lgpd/dpo` e no rodapé da documentação pública ([docs/index.html](docs/index.html)).
3. Estabelecer SLA interno: resposta em até 5 dias úteis.

---

## 9. Transferências internacionais (Art. 33)

O projeto **transfere dados pessoais para fora do Brasil** sempre que:
- `RAG_MODEL` baixa modelo do Hugging Face (servidores nos EUA/UE).
- `VLLM_BASE_URL` aponta para vLLM em **RunPod** (geralmente EUA/Europa) — recebe **prompt + context completos** em cada requisição ([generator.py:177-179](src/rag/generator.py#L177-L179), [generator.py:234](src/rag/generator.py#L234)).
- `RAG_GENERATOR_URL` aponta para BentoML remoto (alternativa configurável).
- Variáveis `HF_TOKEN` / `VLLM_API_KEY` / `HUGGING_FACE_HUB_TOKEN` autenticam contra servidores estrangeiros.

**Hipóteses do Art. 33 aplicáveis:**
- I — Países com nível de proteção adequado (lista da ANPD — atualmente **vazia**).
- II — Cláusulas contratuais específicas (DPA padrão).
- VII — Consentimento específico e destacado do titular.

**Ação P1.**
1. Listar **todos** os destinos internacionais em `GET /lgpd/sharing` (Hugging Face, RunPod, BentoML remoto se aplicável, Langfuse Cloud se usado).
2. Avaliar se a inferência pode ser feita em endpoint nacional (BentoML/vLLM on-prem em provedor BR) para minimizar transferência.
3. Coletar consentimento explícito mencionando os países de destino se a transferência for inevitável.
4. Garantir que `HF_TOKEN` / `VLLM_API_KEY` não enviem payloads contendo PII — aplicar PII scrubbing **antes** da chamada a `VLLM_BASE_URL`.

---

## 10. Plano de retenção e eliminação (Art. 15–16)

**Princípio.** Dado pessoal só pode ser mantido enquanto necessário para a finalidade. Findo isso → eliminação (Art. 16) ou anonimização irreversível.

### Política proposta

| Tipo de dado | Retenção | Justificativa |
|--------------|----------|---------------|
| Documento ingerido em `/ingest` | 90 dias após último acesso, ou imediato se titular revogar | Necessidade operacional do RAG |
| Query do usuário (logs) | 30 dias | Segurança / detecção de fraude (Art. 7º, II) |
| Embedding derivado | Mesmo prazo do documento original | Embedding é dado pessoal (reversível) |
| Métricas agregadas (MLflow) | Indeterminado **se anonimizadas** | Não constitui dado pessoal após agregação |
| Logs de incidente de segurança | 6 meses | Investigação |

**Implementação técnica.**
1. Job agendado (ex.: cron diário) que percorre `metadata` e remove documentos expirados.
2. Migrar de `IndexFlatL2` para `IndexIDMap2` que suporta `remove_ids()`.
3. Eliminar **também** runs MLflow vinculados (`mlflow.delete_run`) — hoje fica órfão.
4. Sobrescrita segura: anonimização preferível à deleção quando há dependência (modelo treinado).

**Ação P1.**

---

## 11. Gestão de incidentes (Art. 48)

**Obrigação.** Comunicar ANPD e o titular **em prazo razoável** (definido pela ANPD) sobre incidentes que causem risco/dano relevante.

### Procedimento

```
1. DETECÇÃO     → alerta automatizado (PII em log, 401/403 anômalo, tráfego /ingest)
2. CONTENÇÃO    → revogar API key, isolar índice, snapshot forense
3. ANÁLISE      → determinar dado afetado, número de titulares, vetor explorado
4. NOTIFICAÇÃO  → Encarregado → ANPD (formulário) + titulares afetados, em <72h
5. REMEDIAÇÃO   → patch, rotação de credenciais, comunicação pública
6. POST-MORTEM  → atualizar OWASP/RED_TEAMING/RIPD
```

**Conteúdo mínimo da notificação ao titular** (Art. 48, §1º):
- Descrição da natureza dos dados afetados.
- Informações sobre os titulares envolvidos.
- Indicação das medidas técnicas e de segurança utilizadas.
- Riscos relacionados.
- Motivos da demora, se houver.
- Medidas adotadas para reverter ou mitigar.

**Ação P0.** Criar `docs/INCIDENT_RESPONSE.md` (playbook). Definir canal de comunicação (e-mail do DPO + form público). Configurar alertas em **Prometheus/Grafana** ([docker-compose.yaml:96-135](docker-compose.yaml#L96-L135)) para detecção automatizada de anomalias.

---

## 12. Roadmap de adequação

Ordem de execução recomendada, alinhada com prioridades de [docs/OWASP.md](docs/OWASP.md) §4.

### Sprint 1 (P0 — bloqueadores legais e técnicos)
- [ ] Designar Encarregado e publicar contato.
- [ ] Política de Privacidade publicada (`GET /lgpd/policy`).
- [ ] Autenticação obrigatória em `/ingest`, `/ingest_mlflow`, `/query`, `/agent`.
- [ ] Integrar `InputGuardrail` e `OutputGuardrail` em [src/serving/app.py](src/serving/app.py).
- [ ] Anonimização de PII na ingestão (Presidio antes do embedding).
- [ ] **PII scrubbing antes do envio ao vLLM remoto** (`VLLM_BASE_URL`).
- [ ] Validação Pydantic: rejeitar campos com palavras-chave de dado sensível (Art. 11).
- [ ] Hardening do stack de observabilidade (Grafana/Langfuse sem defaults, secrets via secret manager).
- [ ] Iniciar RIPD com template ANPD.

### Sprint 2 (P1 — direitos dos titulares + retenção)
- [ ] Vincular documentos a `subject_id`/`tenant_id` na metadata FAISS.
- [ ] Migrar para `IndexIDMap2` (suporta deleção).
- [ ] Endpoints `/lgpd/*` (acesso, correção, eliminação, portabilidade, revogação).
- [ ] Job de retenção (90 dias) com eliminação cascata MLflow + FAISS.
- [ ] Logging estruturado com `request_id` e redaction de PII.

### Sprint 3 (P2 — robustez e governança)
- [ ] Particionamento de índice por tenant.
- [ ] Inferência on-prem para minimizar transferência internacional (Art. 33).
- [ ] Avaliação de viés em recomendações financeiras (Art. 20).
- [ ] Auditoria externa / pen-test executando os 5 cenários de [docs/RED_TEAM_REPORT.md](docs/RED_TEAM_REPORT.md).
- [ ] Treinamento LGPD + IA responsável da equipe.

---

## 13. Checklist de conformidade

Use como auto-avaliação periódica (trimestral).

### Estrutural
- [ ] Encarregado designado e publicado
- [ ] Política de Privacidade pública e atualizada
- [ ] Registro de Operações de Tratamento (Art. 37) atualizado
- [ ] RIPD elaborado e revisado pelo Encarregado
- [ ] Termo de Consentimento exibido antes da coleta
- [ ] DPA assinado com cada operador (Hugging Face, RunPod, BentoML, Langfuse)

### Técnico
- [ ] Autenticação em todos os endpoints (incluindo `/ingest_mlflow`)
- [ ] Anonimização/pseudonimização aplicada na ingestão, no output **e antes do envio à RunPod**
- [ ] Logs sem PII (redaction validada)
- [ ] TLS obrigatório no acesso externo
- [ ] Particionamento de índice por titular/tenant
- [ ] Suporte a deleção real (FAISS migrado)
- [ ] Job de retenção rodando com sucesso (incluindo traces no Langfuse)
- [ ] Stack de observabilidade com secrets gerenciados (sem defaults no compose)
- [ ] Métricas de tempo de resposta a direitos do titular dentro do SLA (15 dias)

### Operacional
- [ ] Canal público para o titular exercer direitos (`/lgpd/me/*` + e-mail DPO)
- [ ] Playbook de incidente testado em *tabletop exercise*
- [ ] Treinamento LGPD da equipe — última edição < 12 meses
- [ ] Auditoria interna anual + pen-test (cenários de [docs/RED_TEAM_REPORT.md](docs/RED_TEAM_REPORT.md))
- [ ] Inventário de transferências internacionais revisado

---

## Referências

- [Lei 13.709/2018 — LGPD (texto integral)](https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm)
- [ANPD — Agência Nacional de Proteção de Dados](https://www.gov.br/anpd/pt-br)
- [Guia ANPD — Tratamento de Dados Pessoais por IA](https://www.gov.br/anpd/pt-br/assuntos/noticias)
- [Guia ANPD — Elaboração de RIPD](https://www.gov.br/anpd/pt-br/documentos-e-publicacoes)
- [Resolução CD/ANPD 2/2022 — Aplicação a agentes de pequeno porte](https://www.gov.br/anpd/pt-br)
- Documentação interna: [docs/OWASP.md](docs/OWASP.md), [docs/RED_TEAM_REPORT.md](docs/RED_TEAM_REPORT.md), [docs/EXPLAINABILITY_FAIRNESS.md](docs/EXPLAINABILITY_FAIRNESS.md)
