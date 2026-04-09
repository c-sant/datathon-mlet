---

## 🔹 Microsoft MLOps Maturity Model – Experiment Management

A dimensão *Experiment Management* avalia como os experimentos de machine learning são organizados, rastreados e comparados.  
Nosso projeto já cobre de forma sólida os níveis 2 e 3.

| Nível | Descrição | O que temos implementado |
|-------|-----------|---------------------------|
| **Stage 1 – Inicial** | Experimentos ad hoc, sem rastreabilidade. | ❌ Não aplicável, já superado. |
| **Stage 2 – Básico** | Registro manual de métricas e parâmetros, pouca padronização. | ✅ Runs separados por framework.<br>✅ Métricas (MAE, RMSE, MAPE) logadas.<br>✅ Parâmetros registrados (ticker, janela, epochs, batch_size, patience). |
| **Stage 3 – Intermediário** | Uso de ferramenta de tracking (MLflow), runs padronizados, comparabilidade sistemática. | ✅ MLflow padronizado.<br>✅ Artefatos registrados (modelos, CSV, gráficos).<br>✅ Benchmark automatizado para comparação entre frameworks.<br>✅ Benchmark com ≥ 3 configurações documentadas (PyTorch, Sklearn, Keras). |
| **Stage 4 – Avançado** | Automação de experimentos, versionamento de datasets, integração com pipelines CI/CD. | ⚠️ Ainda não implementado. Planejado para evolução futura. |
| **Stage 5 – Otimizado** | Dashboards interativos, governança, auditoria e reprodutibilidade total. | ⚠️ Ainda não implementado. Planejado para evolução futura. |

---

📌 **Conclusão:**  
Estamos **bem estruturados até os níveis 2 e 3**, garantindo rastreabilidade, reprodutibilidade e comparabilidade dos experimentos.

---

## 🔹 Sistema RAG (Retrieval-Augmented Generation)

Para detalhes completos sobre o sistema de RAG implementado, consulte a documentação específica:  
**[src/rag/README.md](src/rag/README.md)**

Para testar localmente, veja o guia rápido:  
**[TESTE_LOCAL.md](TESTE_LOCAL.md)**

## Teste local rápido

Use o arquivo de dependências local para instalar o ambiente correto:

```powershell
cd "C:\Users\cabri\Documents\Fiap\challenge5\datathon-grupo-05"
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements_local.txt -f https://download.pytorch.org/whl/torch_stable.html
```

### Opção 1: Teste Offline (Sem Download de Modelos) ⭐ RECOMENDADO
Mais rápido e com respostas contextualmente corretas:
```powershell
python test_rag_offline.py
```

### Opção 2: Teste Completo com Respostas Inteligentes em Português ⭐ NOVO PADRÃO
Agora usa respostas simuladas inteligentes (sempre funciona, português perfeito):
```powershell
python run_local.py
```

### Opção 3: Teste com Modelo Real (facebook/opt-1.3b)
Para desenvolvimento avançado com modelo real:
```powershell
set RAG_MODEL=facebook/opt-1.3b
python run_local.py
```

### Opção 4: Teste com Modelo Rápido (distilgpt2)
Modelo leve, mas pode gerar texto estranho:
```powershell
set RAG_MODEL=distilgpt2
python run_local.py
```

Se `run_local.py` falhar com problema de rede, veja [TESTE_LOCAL.md](TESTE_LOCAL.md#erro-429-too-many-requests-do-hugging-face) para alternativas.

> **Nota sobre modelos**: 
> - `simulated` (padrão): Respostas perfeitas em português, sempre funciona
> - `facebook/opt-1.3b`: Melhor qualidade geral
> - `distilgpt2`: Rápido, mas gera texto estranho com prompts em PT
> - `test_rag_offline.py`: Sempre funciona, respostas contextuais inteligentes

Se o PowerShell bloquear a ativação do virtualenv, execute:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

O RAG permite consultas contextuais baseadas em dados fornecidos, integrando ingestão dinâmica, busca vetorial (FAISS) e geração de respostas via LLM (BentoML/vLLM). 