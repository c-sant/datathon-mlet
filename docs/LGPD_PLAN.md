# Plano LGPD Aplicado — Datathon MLET (RAG + FastAPI)

Plano de conformidade com a **Lei Geral de Proteção de Dados Pessoais (Lei nº 13.709/2018)** aplicado ao caso real deste projeto: um sistema **RAG** que combina FastAPI, FAISS, MLflow/DVC e LLM servido via BentoML+vLLM (RunPod ou local).

Este plano parte da arquitetura existente (não é um template genérico): endereça o que já está implementado em [src/security/guardrails.py](src/security/guardrails.py) e [src/security/pii_detection.py](src/security/pii_detection.py), identifica lacunas concretas e define ações priorizadas.

> **Complementaridade**: este documento trata de **dados pessoais** (LGPD). Para ameaças técnicas do LLM ver [docs/OWASP.md](docs/OWASP.md). Muitos controles se sobrepõem — aqui olhamos pelo ângulo jurídico-regulatório.

---

## 1. Contexto e papéis (Art. 5º, VI–VIII)

| Papel LGPD | Atribuição no projeto |
|------------|-----------------------|
| **Controlador** | Entidade que decide finalidade e meios (ex.: instituição que operar o RAG em produção). Na fase acadêmica/datathon, o grupo de alunos. |
| **Operador** | Infraestrutura de terceiros: **RunPod** (GPU), **Hugging Face** (download de modelos/tokenizers), **Docker Hub** (imagens). |
| **Encarregado (DPO)** | A nomear antes de qualquer uso produtivo. Papel pendente. |
| **Titular** | Qualquer pessoa cujos dados apareçam no corpus ingerido ([data/ingest.py](data/ingest.py)) ou nos prompts enviados a `/predict`. |

**Ação P0**: formalizar o controlador e nomear DPO antes de publicar o serviço fora do ambiente de estudo. Registrar canal público (e-mail/URL) conforme Art. 41, §1º.

---

## 2. Inventário de dados pessoais (ROPA — Art. 37)

Mapeamento das entradas/saídas do sistema onde dados pessoais podem trafegar:

| # | Ponto do fluxo | Arquivo(s) | Categorias potenciais | Sensível? |
|---|----------------|------------|------------------------|-----------|
| A | Documentos ingeridos → FAISS | [data/ingest.py](data/ingest.py), [src/rag/embedding.py](src/rag/embedding.py) | Nome, CPF, CNPJ, e-mail, telefone, CEP, IP, cartão | Possível dado financeiro |
| B | Prompt do usuário (`/predict`) | [app/main.py](app/main.py), [src/rag/retriever.py](src/rag/retriever.py) | Qualquer PII digitada pelo usuário | Depende do uso |
| C | Contexto recuperado + prompt → LLM | [src/rag/generator.py](src/rag/generator.py), vLLM remoto (RunPod) | Herda de A e B | **Transferência internacional** |
| D | Resposta gerada pelo LLM | [src/rag/generator.py](src/rag/generator.py) | Risco de regurgitação de PII do corpus | Sim |
| E | Logs de aplicação | logging padrão em `app/` e `src/` | Prompt cru pode estar nos logs | **Alto risco** |
| F | Artefatos MLflow | [mlruns/](mlruns/) | Samples de teste podem conter PII | Médio |
| G | Versionamento DVC | `dvc.yaml`, cache DVC | Cópia dos dados ingeridos | Médio |

**Ação P0**: preencher este inventário com volumes, retenção e origem antes de cada release. Versionar o ROPA neste repositório (`docs/ropa.md`) e atualizá-lo em todo PR que toque em `data/`, `src/rag/` ou `src/security/`.

---

## 3. Bases legais (Art. 7 e 11)

Dependem do caso de uso final. Recomendações:

| Cenário | Base legal sugerida | Observações |
|---------|--------------------|-------------|
| Protótipo/datathon com dados sintéticos ou públicos | **Art. 7º, IV** (estudos por órgão de pesquisa) e **Art. 4º, II, a** (fins acadêmicos) | Exige ausência de PII ou anonimização — ver §4. |
| Produto comercial com clientes cadastrados | **Art. 7º, V** (execução de contrato) + **IX** (legítimo interesse) para melhorias | Legítimo interesse exige teste de balanceamento documentado. |
| Uso de dados financeiros (perfil de investidor) | **Art. 7º, I** (consentimento explícito) + regulação CVM/BACEN | Consentimento granular e revogável. |
| Dado sensível (saúde, biometria, político) — **evitar** | Art. 11 — consentimento específico e destacado | Hoje nada no pipeline justifica coletar dado sensível. Manter exclusão por desenho. |

**Ação P1**: escolher a base legal antes de qualquer ingestão de dados reais e registrá-la no frontmatter do dataset em DVC/MLflow.

---

## 4. Princípios aplicados ao pipeline (Art. 6)

### 4.1. Finalidade, adequação e necessidade

- `/predict` hoje retorna um label baseado no tamanho do texto ([app/main.py:32-41](app/main.py#L32-L41)) — **não precisa** de PII. Recusar prompts com PII na entrada por padrão é a aplicação direta do princípio da necessidade.
- **Ação P1**: rodar [src/security/pii_detection.py](src/security/pii_detection.py) no pipeline de ingestão e **não indexar** chunks com PII que não seja estritamente necessária para a tarefa.

### 4.2. Livre acesso, qualidade e transparência

- **Ação P1**: expor endpoint `/privacy` ou página estática declarando finalidade, categorias de dado, retenção e canal do DPO (Art. 9º).
- Documentar, em [README.md](README.md), que o modelo pode errar ("alucinar") — relevante para transparência quando a resposta embasar decisão do titular.

### 4.3. Segurança e prevenção (Art. 6º, VII–VIII)

Já implementado:
- **Sanitização de PII** pós-LLM via Presidio em [src/security/guardrails.py:64-83](src/security/guardrails.py#L64-L83) (entidades `PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `BR_CPF`).
- **Detector regex** com CPF, CNPJ, e-mail, telefone, cartão, CEP, IP em [src/security/pii_detection.py:23-31](src/security/pii_detection.py#L23-L31).
- **Auditoria de acesso**: [src/security/pii_detection.py:93-118](src/security/pii_detection.py#L93-L118) (`audit_log_pii`) — estrutura pronta, falta plugar no fluxo real.
- **Bloqueio de prompt injection**: [src/security/guardrails.py:15-28](src/security/guardrails.py#L15-L28).

Lacunas (ver §6).

### 4.4. Não discriminação (Art. 6º, IX)

Ativos financeiros recomendados não podem discriminar por raça, gênero ou origem. Mitigações:
- **Ação P2**: incluir teste de fairness no benchmark ([evaluation/benchmark.py](evaluation/benchmark.py)) com queries contrastivas (mesma pergunta, variação demográfica).

### 4.5. Responsabilização e prestação de contas

- DVC + MLflow fornecem trilha técnica. Falta a trilha **jurídica**: quem autorizou qual dataset, com qual base legal.
- **Ação P1**: adicionar campo `legal_basis` em `params.yaml` e propagar como tag MLflow.

---

## 5. Direitos do titular (Art. 18) — implementação técnica

| Direito | Como atender no sistema atual | Status |
|---------|-------------------------------|--------|
| Confirmação e acesso (I, II) | Exportar do FAISS + MLflow os registros associados ao titular | **Pendente** — requer campo `subject_id` no metadata dos chunks |
| Correção (III) | Reingestão via [data/ingest.py](data/ingest.py) com dado corrigido + rebuild do índice | Parcial — processo manual |
| Anonimização/bloqueio/eliminação (IV) | Remover chunks por `subject_id` no FAISS + purge do cache DVC | **Pendente** — FAISS atual não suporta delete nativo eficiente |
| Portabilidade (V) | Exportar chunks + metadados em JSON | **Pendente** |
| Eliminação após consentimento (VI) | Script de purge cross-sistema (FAISS, DVC, MLflow, logs) | **Pendente** |
| Informação sobre compartilhamento (VII) | Declarar RunPod/HuggingFace no aviso de privacidade | **Pendente** |
| Revisão de decisão automatizada (§ Art. 20) | Adicionar disclaimer em respostas do `/predict` e canal para revisão humana | **Pendente** |

**Ação P0**: adicionar campo obrigatório `subject_id` (ou hash pseudonimizado) no `metadata` de cada chunk em [src/rag/embedding.py](src/rag/embedding.py). Sem isso, os direitos dos Art. 18, IV/V/VI são inviáveis tecnicamente.

**Ação P1**: implementar script `scripts/lgpd_erase.py` que aceita `subject_id` e remove:
1. Chunks do FAISS (reconstruindo o índice ou usando `IndexIDMap`).
2. Registros no DVC cache.
3. Runs MLflow associados.
4. Entradas nos logs (com truncamento + rotação).

---

## 6. Medidas de segurança (Art. 46) — gaps e ações

| Medida | Status | Gap | Ação |
|--------|--------|-----|------|
| Pseudonimização no corpus | Parcial (sanitizer existe, não integrado na ingestão) | Chunks indexados com PII crua | **P0**: chamar `detect_and_mask_pii` em `data/ingest.py` antes de vetorizar |
| Criptografia em trânsito | Ausente | `/predict` sem HTTPS; chamada a vLLM via HTTP | **P0**: TLS no reverse proxy; `VLLM_BASE_URL` exige `https://` |
| Criptografia em repouso | Ausente | FAISS, DVC cache e logs em disco plain | **P1**: volumes Docker criptografados; `mlruns/` em storage com KMS |
| Controle de acesso | Ausente | `/predict` aberto | **P0**: API key + rate limit (também endereça OWASP LLM10) |
| Logs com PII | **Crítico** | [src/rag/generator.py](src/rag/generator.py) e `logger.warning` em [src/security/guardrails.py:46](src/security/guardrails.py#L46) podem logar prompt com PII (`user_input[:100]`) | **P0**: passar sempre por `detect_and_mask_pii` antes de logar |
| Retenção e expurgo | Ausente | Dados ficam indefinidamente no FAISS/DVC | **P1**: política de retenção em `params.yaml` + cron de expurgo |
| Segregação de ambientes | Parcial (compose dev vs. RunPod) | Sem separação prod/staging/dev | **P2**: três compose files + tags distintas MLflow |
| Backup e recuperação | Ausente | Sem plano formal | **P2**: snapshot do FAISS + DVC remote |
| Gestão de acessos | Ausente | Qualquer dev tem acesso total | **P1**: IAM mínimo no RunPod, credenciais MLflow por pessoa |

---

## 7. Transferência internacional de dados (Art. 33) — ponto crítico

O caminho validado em [CLOUD_RUNPOD_QUICKSTART.md](CLOUD_RUNPOD_QUICKSTART.md) envia prompts (potencialmente com PII) ao **vLLM no RunPod**, cujas GPUs podem estar **fora do Brasil**. Isso caracteriza transferência internacional e **exige uma das hipóteses do Art. 33**:

| Hipótese | Viabilidade neste projeto |
|----------|---------------------------|
| País com nível de proteção adequado (ANPD) | RunPod não é país — depende da região de GPU. Conferir antes de ir a prod. |
| Cláusulas contratuais padrão | Viável — exige contrato com RunPod espelhando LGPD. |
| Consentimento específico | Operacionalmente caro no fluxo RAG. |
| Necessidade para contrato | Possível se o titular for o próprio usuário. |

**Ação P0**:
1. **Sanitizar o prompt antes** de sair para o vLLM (chamar `detect_and_mask_pii` em [src/rag/generator.py](src/rag/generator.py) antes do POST). Assim, o que trafega para fora já é pseudonimizado e reduz-se o escopo da transferência.
2. Documentar a região da GPU contratada no RunPod e decidir base do Art. 33.
3. Alternativa: caminho CPU local ([docker/docker-compose.yml](docker/docker-compose.yml)) para cargas com PII, caminho GPU remota só para cargas sanitizadas.

---

## 8. Relatório de Impacto à Proteção de Dados (RIPD / DPIA) — Art. 38

Exigível quando houver **alto risco** aos titulares. Este sistema provavelmente qualifica por: (a) tratamento automatizado com efeito jurídico (recomendação financeira); (b) uso de IA sobre dados que podem incluir perfil financeiro; (c) transferência internacional.

**Ação P1**: produzir `docs/RIPD.md` cobrindo:
- Descrição dos tratamentos (reaproveitar §2).
- Necessidade e proporcionalidade.
- Riscos aos direitos dos titulares (reaproveitar §5).
- Medidas de mitigação (reaproveitar §6).
- Consulta ao DPO.

---

## 9. Gestão de incidentes (Art. 48)

Obrigação de comunicar ANPD e titulares em **prazo razoável** (ANPD vem recomendando até 3 dias úteis).

**Ação P1**: playbook `docs/IR_playbook.md` com:
1. Detecção (alertas sobre `OutputGuardrail` removendo PII, picos anômalos em `/predict`).
2. Contenção (revogar API keys, rollback do último `ingest` via DVC).
3. Análise de impacto (quais `subject_id` afetados — depende da ação P0 da §5).
4. Notificação ANPD + titulares.
5. Postmortem com atualização deste plano.

Integrar com o workflow já existente em [.github/workflows/ci-cd.yml](.github/workflows/ci-cd.yml) para criação automática de issue em incidentes detectados.

---

## 10. Ciclo de vida dos dados

```
 Coleta ───────► Ingestão ───────► Indexação ───────► Uso ───────► Retenção ───────► Expurgo
   │                │                   │               │              │                │
[fonte         [data/ingest    [FAISS + DVC]     [/predict →      [política em       [scripts/
 declarada]    + PII mask]                       vLLM + PII         params.yaml]     lgpd_erase.py]
                                                  mask pré-envio]
```

Pontos de controle:
- **Entrada**: valida consentimento/base legal; roda PII mask.
- **Índice**: metadata inclui `subject_id`, `legal_basis`, `retention_until`.
- **Uso**: guardrail de entrada + saída; log sanitizado; telemetria sem PII.
- **Saída**: job periódico remove chunks cuja `retention_until` expirou.

---

## 11. Treinamento e cultura

- **Ação P2**: README do módulo [src/security/](src/security/) com exemplos de uso dos guardrails para novos contribuidores.
- **Ação P2**: checklist LGPD em `.github/pull_request_template.md` para PRs que toquem `data/`, `src/rag/` ou `src/security/`.

---

## 12. Roadmap priorizado

### P0 — antes de qualquer exposição real
1. **Sanitização de PII na ingestão** — integrar [src/security/pii_detection.py](src/security/pii_detection.py) em [data/ingest.py](data/ingest.py).
2. **Sanitização antes do envio ao vLLM remoto** — reduz escopo da transferência internacional.
3. **Campo `subject_id` no metadata dos chunks** — viabiliza direitos do Art. 18.
4. **Logs sanitizados** — remover `user_input[:100]` cru de [src/security/guardrails.py:46](src/security/guardrails.py#L46).
5. **API key + HTTPS** em `/predict`.
6. **Nomear DPO e controlador; publicar canal** (Art. 41).

### P1 — antes de produção
7. Script `scripts/lgpd_erase.py` (direito de eliminação).
8. Aviso de privacidade público e endpoint `/privacy`.
9. RIPD formalizado em `docs/RIPD.md`.
10. Playbook de incidentes.
11. Política de retenção em `params.yaml` + job de expurgo.
12. `legal_basis` como tag em MLflow.

### P2 — maturidade
13. Testes de fairness no benchmark.
14. Criptografia em repouso (volumes, mlruns/).
15. Checklist LGPD em PR template.
16. Segregação prod/staging/dev completa.

---

## 13. Rastreamento no código

| Arquivo | Função LGPD |
|---------|-------------|
| [src/security/pii_detection.py](src/security/pii_detection.py) | Detecção/mascaramento + auditoria (Art. 46, 37) |
| [src/security/guardrails.py](src/security/guardrails.py) | Sanitização de saída e filtro de input (Art. 46) |
| [data/ingest.py](data/ingest.py) | Ponto de aplicação da minimização (Art. 6, III) — **gap atual** |
| [src/rag/generator.py](src/rag/generator.py) | Ponto de saneamento pré-transferência (Art. 33) — **gap atual** |
| [app/main.py](app/main.py) | Controle de acesso e rate limit — **gap atual** |
| [tests/test_guardrails.py](tests/test_guardrails.py) | Evidência de mitigação (Art. 50, §2º, II) |

---

## Referências

- Lei nº 13.709/2018 (LGPD) — https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm
- ANPD — Guia de Segurança da Informação para Agentes de Tratamento — https://www.gov.br/anpd/
- ANPD — Resolução CD/ANPD nº 2/2022 (tratamento pequenos agentes).
- ENISA — AI Cybersecurity Challenges (para cruzamento LGPD ↔ riscos IA).
- [docs/OWASP.md](docs/OWASP.md) — ameaças técnicas do LLM que este plano endereça pelo ângulo jurídico.
