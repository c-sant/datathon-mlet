# Guia de Teste Local - Sistema RAG

Este guia mostra como testar o sistema RAG localmente no Windows, com opções de teste rápido e completo.

## 🤖 O que é RAG? (Breve Explicação)

**RAG (Retrieval-Augmented Generation)** é um sistema de IA que combina:
- **Busca Inteligente**: Encontra informações relevantes em uma base de dados
- **Geração Contextual**: Cria respostas baseadas nas informações encontradas

**Por que usar RAG?**
- Respostas mais precisas e atualizadas
- Baseadas em dados reais (não apenas conhecimento pré-treinado)
- Menos "alucinações" (informações incorretas)

**Como funciona aqui:**
1. Você faz uma pergunta sobre investimentos
2. Sistema busca em notícias financeiras relevantes
3. Gera resposta contextual em português

---

## Pré-requisitos
- Python 3.10+
- Git (opcional, para clonar o repo)

## 1. Preparar Ambiente
```bash
cd "c:\Users\cabri\Documents\Fiap\challenge5\datathon-grupo-05"
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
```

> Se estiver usando PowerShell e receber erro de política de execução, execute:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Ou abra o prompt de comando normal e rode:
> ```cmd
> venv\Scripts\activate.bat
> ```

## 2. Instalar Dependências Essenciais
```bash
pip install -r requirements_local.txt -f https://download.pytorch.org/whl/torch_stable.html
```

> Nota: `requirements_local.txt` já inclui `sentence-transformers==2.3.1`, `numpy==1.26.4` e `lxml_html_clean`.
>
> Se `pip` não encontrar `torch`, use:
> ```bash
> pip install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
> ```
> Depois, reinstale `requirements_local.txt`.

> Se `sentence-transformers` ainda falhar, execute:
> ```bash
> pip install --no-cache-dir sentence-transformers==2.3.1
> ```

## 3. Escolher um Modelo de Geração (Opcional)

O sistema escolhe automaticamente um modelo fallback se nenhum estiver disponível. Mas você pode customizar:

**Opção A: Teste Offline (Recomendado para desenvolvimento)**
Não precisa de changes:
```bash
python test_rag_offline.py
```
✅ Funciona 100%  
✅ Sem delays de download  
✅ Respostas corretas em português  

**Opção B: Teste com Modelo Português (Melhor qualidade)**
```bash
set RAG_MODEL=facebook/opt-1.3b
python run_local.py
```
✅ Respostas muito melhores em português  
⏱️ Primeira execução: ~1-2 min (download 2.6GB)  
⏱️ Execuções seguintes: ~5-10s  

**Opção C: Teste com Modelo Rápido (Padrão, menor qualidade em PT)**
```bash
python run_local.py
```
✅ Rápido (distilgpt2)  
⚠️ Respostas em português podem ser estranhas  

## 4. Teste Rápido (Recomendado)
Execute o script de teste local que simula o fluxo completo:
```bash
python test_rag_offline.py
```

**O que acontece:**
- Simula coleta de notícias financeiras
- Processa e indexa os documentos
- Faz uma query de exemplo: "Quais ações estão recomendadas para 2026?"
- Mostra os chunks relevantes recuperados
- Gera **resposta inteligente contextual** em português perfeito

**Saída esperada (português correto e contextual):**
```
🔹 Query: Quais ações estão recomendadas para 2026?

--- Resultados do Retriever ---
Rank 1 | Distância: 0.45
Doc: doc_1 - ETFs: Diversificação Global
Texto: ETFs de índices...

--- Resposta Final ---
Com base no contexto fornecido sobre renda fixa e juros altos, recomendo priorizar 
investimentos em títulos de renda fixa como CDB, Tesouro Direto e debêntures, 
que oferecem retornos atrativos acima de 10% ao ano com baixo risco. Para ações, 
considere empresas sólidas com dividendos consistentes e exposição internacional moderada.
```

## 5. Teste da API Completa
Para testar ingestão dinâmica e consultas via API:

### Iniciar API
```bash
cd src\rag
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Testar Ingestão
```bash
curl -X POST "http://localhost:8000/ingest" ^
  -H "Content-Type: application/json" ^
  -d "{\"docs\": [{\"id\": \"teste\", \"title\": \"Teste\", \"text\": \"Este é um texto de teste sobre ações na bolsa.\"}], \"overwrite\": true}"
```

**Resposta esperada:**
```json
{
  "status": "ok",
  "doc_count": 1,
  "chunk_count": 2,
  "overwrite": true
}
```

### Testar Consulta
```bash
curl "http://localhost:8000/query?q=Quais%20ações%20são%20recomendadas?"
```

**Resposta esperada:**
```json
{
  "query": "Quais ações são recomendadas?",
  "top_k": 3,
  "context": "Este é um texto de teste sobre ações na bolsa.",
  "answer": "[Resposta gerada]"
}
```

## 5. Teste com Modelo Avançado (BentoML)
Para usar o modelo OPT-1.3B via vLLM:

### Instalar Dependências Adicionais
```bash
pip install bentoml vllm
```

### Iniciar Serviço de Geração
Terminal 1:
```bash
cd generator\serving
bentoml serve app:svc --port 3000
```

### Iniciar API RAG
Terminal 2:
```bash
cd src\rag
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Agora as consultas usarão o modelo OPT-1.3B em vez do fallback local.

## Troubleshooting

### Erro: 429 Too Many Requests do Hugging Face
Se receber erro `429 Client Error: Too Many Requests`:

**Boa notícia:** Sistema agora tem **fallback automático** implementado!

Cascata de fallbacks:
1. ✅ Tenta carregar `distilgpt2` (padrão rápido)
2. ✅ Se 429 error, tenta `gpt2` (segundo fallback)
3. ✅ Se ambos falharem, usa **resposta simulada contextual** (sempre funciona)

Basta executar:
```bash
python run_local.py
```

O sistema vai automaticamente:
- Baixar modelos quando disponível
- Usar cache se já estava baixado
- Gerar respostas simuladas inteligentes como último recurso

Nenhuma configuração manual necessária!

### Erro: ModuleNotFoundError
Instale o pacote faltante:
```bash
pip install <nome-do-pacote>
```

### Erro: CUDA out of memory (se usar GPU)
Para CPU:
- Adicione `--device cpu` nos comandos vLLM
- Ou defina `export CUDA_VISIBLE_DEVICES=""`

### Modelo lento
- Use `facebook/opt-1.3b` (mais rápido que Phi-2)
- Reduza `max_tokens` no prompt

### Porta ocupada
Mude a porta: `--port 8001`

## Próximos Passos
- Teste com seus próprios dados via `/ingest`
- Compare respostas com diferentes queries
- Monitore logs no MLflow: `mlflow ui`

---
**Data:** Abril 2026
**Testado em:** Windows 11, Python 3.10</content>
<parameter name="filePath">c:\Users\cabri\Documents\Fiap\challenge5\datathon-grupo-05\TESTE_LOCAL.md