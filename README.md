---

## ðŸ”¹ Microsoft MLOps Maturity Model â€“ Experiment Management

A dimensÃ£o *Experiment Management* avalia como os experimentos de machine learning sÃ£o organizados, rastreados e comparados.  
Nosso projeto jÃ¡ cobre de forma sÃ³lida os nÃ­veis 2 e 3.

| NÃ­vel | DescriÃ§Ã£o | O que temos implementado |
|-------|-----------|---------------------------|
| **Stage 1 â€“ Inicial** | Experimentos ad hoc, sem rastreabilidade. | âŒ NÃ£o aplicÃ¡vel, jÃ¡ superado. |
| **Stage 2 â€“ BÃ¡sico** | Registro manual de mÃ©tricas e parÃ¢metros, pouca padronizaÃ§Ã£o. | âœ… Runs separados por framework.<br>âœ… MÃ©tricas (MAE, RMSE, MAPE) logadas.<br>âœ… ParÃ¢metros registrados (ticker, janela, epochs, batch_size, patience). |
| **Stage 3 â€“ IntermediÃ¡rio** | Uso de ferramenta de tracking (MLflow), runs padronizados, comparabilidade sistemÃ¡tica. | âœ… MLflow padronizado.<br>âœ… Artefatos registrados (modelos, CSV, grÃ¡ficos).<br>âœ… Benchmark automatizado para comparaÃ§Ã£o entre frameworks.<br>âœ… Benchmark com â‰¥ 3 configuraÃ§Ãµes documentadas (PyTorch, Sklearn, Keras). |
| **Stage 4 â€“ AvanÃ§ado** | AutomaÃ§Ã£o de experimentos, versionamento de datasets, integraÃ§Ã£o com pipelines CI/CD. | âš ï¸ Ainda nÃ£o implementado. Planejado para evoluÃ§Ã£o futura. |
| **Stage 5 â€“ Otimizado** | Dashboards interativos, governanÃ§a, auditoria e reprodutibilidade total. | âš ï¸ Ainda nÃ£o implementado. Planejado para evoluÃ§Ã£o futura. |

---

ðŸ“Œ **ConclusÃ£o:**  
Estamos **bem estruturados atÃ© os nÃ­veis 2 e 3**, garantindo rastreabilidade, reprodutibilidade e comparabilidade dos experimentos.

---

## ðŸ”¹ Sistema RAG (Retrieval-Augmented Generation)

### ðŸ“– O que Ã© RAG?

**Retrieval-Augmented Generation (RAG)** Ã© uma arquitetura de IA que combina **recuperaÃ§Ã£o de informaÃ§Ã£o** com **geraÃ§Ã£o de texto**, permitindo que modelos de linguagem respondam com base em dados externos atualizados, em vez de apenas seu conhecimento prÃ©-treinado.

#### ðŸŽ¯ **Conceito TeÃ³rico:**
- **LimitaÃ§Ã£o dos LLMs tradicionais**: Modelos como GPT sÃ£o treinados em dados atÃ© uma data especÃ­fica e podem "alucinar" informaÃ§Ãµes desatualizadas ou incorretas
- **SoluÃ§Ã£o RAG**: Integra uma base de conhecimento externa que Ã© consultada em tempo real para fornecer contexto relevante
- **Vantagem**: Respostas mais precisas, atualizadas e fundamentadas em dados verificÃ¡veis

#### âš™ï¸ **Arquitetura TÃ©cnica:**
```
1. ðŸ“¥ INGESTÃƒO: Documentos â†’ Chunking â†’ Embeddings â†’ VetorizaÃ§Ã£o (FAISS)
2. ðŸ” RETRIEVAL: Query â†’ Embedding â†’ Busca semÃ¢ntica â†’ Top-K chunks relevantes  
3. ðŸ¤– GENERATION: Query + Contexto â†’ LLM â†’ Resposta contextualizada
```

**Componentes principais:**
- **Embedder**: SentenceTransformers (all-MiniLM-L6-v2) para criar representaÃ§Ãµes vetoriais
- **Vector Store**: FAISS com Ã­ndice L2 para busca eficiente O(log n)
- **Retriever**: Busca semÃ¢ntica por similaridade de cosseno
- **Generator**: LLM (BentoML/vLLM ou fallback local) para sÃ­ntese da resposta

#### ðŸŽ¨ **Fluxo de ExecuÃ§Ã£o:**
```
Query: "Quais aÃ§Ãµes sÃ£o recomendadas para 2026?"

1. Query â†’ Embedding vector
2. Busca nos 3 chunks mais similares no FAISS
3. ConcatenaÃ§Ã£o: Query + Contexto relevante
4. GeraÃ§Ã£o: "Com base no contexto sobre renda fixa e juros altos..."
```

Para detalhes completos sobre o sistema de RAG implementado, consulte a documentaÃ§Ã£o especÃ­fica:  
**[src/rag/README.md](src/rag/README.md)**

Para testar localmente, veja o guia rÃ¡pido:  
**[TESTE_LOCAL.md](TESTE_LOCAL.md)**

## Teste local rÃ¡pido

Use o arquivo de dependÃªncias local para instalar o ambiente correto:

```powershell
cd "C:\Users\cabri\Documents\Fiap\challenge5\datathon-grupo-05"
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements_local.txt -f https://download.pytorch.org/whl/torch_stable.html
```

### OpÃ§Ã£o 1: Teste Offline (Sem Download de Modelos) â­ RECOMENDADO
Mais rÃ¡pido e com respostas contextualmente corretas:
```powershell
python test_rag_offline.py
```

### OpÃ§Ã£o 2: Teste Completo com Respostas Inteligentes em PortuguÃªs â­ NOVO PADRÃƒO
Agora usa respostas simuladas inteligentes (sempre funciona, portuguÃªs perfeito):
```powershell
python run_local.py
```

### OpÃ§Ã£o 3: Teste com Modelo Real (facebook/opt-1.3b)
Para desenvolvimento avanÃ§ado com modelo real:
```powershell
set RAG_MODEL=facebook/opt-1.3b
python run_local.py
```

### OpÃ§Ã£o 4: Teste com Modelo RÃ¡pido (distilgpt2)
Modelo leve, mas pode gerar texto estranho:
```powershell
set RAG_MODEL=distilgpt2
python run_local.py
```

Se `run_local.py` falhar com problema de rede, veja [TESTE_LOCAL.md](TESTE_LOCAL.md#erro-429-too-many-requests-do-hugging-face) para alternativas.

> **Nota sobre modelos**: 
> - `simulated` (padrÃ£o): Respostas perfeitas em portuguÃªs, sempre funciona
> - `facebook/opt-1.3b`: Melhor qualidade geral
> - `distilgpt2`: RÃ¡pido, mas gera texto estranho com prompts em PT
> - `test_rag_offline.py`: Sempre funciona, respostas contextuais inteligentes

Se o PowerShell bloquear a ativaÃ§Ã£o do virtualenv, execute:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

O RAG permite consultas contextuais baseadas em dados fornecidos, integrando ingestÃ£o dinÃ¢mica, busca vetorial (FAISS) e geraÃ§Ã£o de respostas via LLM (BentoML/vLLM).

## Stack Docker

### Caminho quantizado na nuvem (RunPod) âœ… VALIDADO

O requisito `LLM servido via API com quantizacao aplicada` e atendido por este caminho:

- `vLLM` com `--quantization awq` rodando em GPU NVIDIA no RunPod
- `BentoML` local (porta `3004`) apontando para `VLLM_BASE_URL` remoto
- Modelo: `Qwen/Qwen2.5-0.5B-Instruct-AWQ`
- Compose dedicado: `docker/docker-compose.bento.remote.yml`

Guia completo de setup e comandos validados: `CLOUD_RUNPOD_QUICKSTART.md`

### Caminho suportado nesta maquina

O stack CPU estavel continua sendo o caminho suportado para servir a API localmente:

```powershell
cd "C:\Users\cabri\Documents\Fiap\challenge5\datathon-grupo-05"
docker compose -f docker/docker-compose.yml up -d
```

Estado verificado neste ambiente:
- `vllm` saudavel em `http://localhost:8001`
- `bentoml` ativo em `http://localhost:3000`

Esse compose usa o modelo leve `facebook/opt-125m` e preserva o fluxo API BentoML + vLLM sem depender de instrucoes AVX512.
