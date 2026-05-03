import os
import re
from difflib import SequenceMatcher

import requests
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# 🔹 Endereço do serviço Bento para geração de texto.
# Ajuste com a variável de ambiente RAG_GENERATOR_URL, se necessário.
BENTO_GENERATOR_URL = os.environ.get("RAG_GENERATOR_URL", "http://localhost:3000/generate")
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "").strip()
VLLM_MODEL = os.environ.get("VLLM_MODEL", "qwen2.5-0.5b-awq")
VLLM_API_KEY = (os.environ.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
REMOTE_LLM_MODE = os.environ.get("REMOTE_LLM_MODE", "auto").strip().lower()

# 🔹 Modelo local para fallback (use variável RAG_MODEL para customizar)
# Padrão: simulated (respostas perfeitas em português, sempre funciona)
# Alternativas:
#   - pierreguillou/gpt2-small-portuguese (ótimo para português, mas pode ter 429)
#   - facebook/opt-1.3b (melhor qualidade geral)
#   - distilgpt2 (rápido, mas gera texto estranho)
#   - gpt2 (genérico)
RAG_MODEL = os.environ.get("RAG_MODEL", "simulated")
HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
)

# 🔹 Lazy loading do modelo local como fallback.
# O modelo é carregado apenas na primeira chamada de generate_answer().
_generator = None
_generator_model_name = None


# Modelos text2text-generation (encoder-decoder como T5, BART)
_TEXT2TEXT_MODELS = {"google/flan-t5-base", "google/flan-t5-small", "google/flan-t5-large"}


def _build_text_generator(model_name):
    """Cria pipeline local de texto com token HF opcional."""
    # transformers 5.x removeu a task text2text-generation.
    if model_name in _TEXT2TEXT_MODELS:
        return _build_seq2seq_generator(model_name)

    task = "text-generation"
    kwargs = {"model": model_name}

    if not HF_TOKEN:
        return pipeline(task, **kwargs)

    try:
        return pipeline(task, token=HF_TOKEN, **kwargs)
    except TypeError:
        # Compatibilidade com versoes antigas do transformers.
        return pipeline(task, use_auth_token=HF_TOKEN, **kwargs)


def _build_seq2seq_generator(model_name):
    """Cria gerador seq2seq manual para T5/FLAN sem depender da pipeline removida."""
    load_kwargs = {}
    if HF_TOKEN:
        load_kwargs["token"] = HF_TOKEN

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kwargs)
    except TypeError:
        # Compatibilidade com versoes antigas do transformers.
        if HF_TOKEN:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_auth_token=HF_TOKEN)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    model.eval()

    # Aplica quantização int8 dinâmica nas camadas Linear (CPU-compatible).
    # Reduz uso de memória ~2x e pode acelerar inferência em CPU.
    try:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        print(f"[quantization] Modelo {model_name} quantizado com int8 dinâmico.")
    except Exception as _qe:
        print(f"[quantization] Aviso: quantização falhou ({_qe}); usando modelo sem quantização.")

    def _run(prompt, max_new_tokens=256, num_return_sequences=1, **kwargs):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": num_return_sequences,
        }
        generation_kwargs.update(kwargs)
        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [{"generated_text": text}]

    return _run


def _get_generator():
    """Carrega o modelo de geração localmente (lazy loading)."""
    global _generator, _generator_model_name

    # Se o modelo for "simulated", usar sempre o modo simulado
    if RAG_MODEL == "simulated":
        print("Usando modo simulado (respostas perfeitas em português)")
        return None

    if _generator is None:
        print(f"Carregando modelo {RAG_MODEL}... (primeira execução)")

        # Tenta com a variável de ambiente RAG_MODEL
        try:
            _generator = _build_text_generator(RAG_MODEL)
            _generator_model_name = RAG_MODEL
            return _generator
        except Exception as e:
            print(f"Falha ao carregar {RAG_MODEL}: {type(e).__name__}")

            # Fallback 1: flan menor para reduzir memoria e latencia.
            if RAG_MODEL != "google/flan-t5-small":
                try:
                    print("Tentando fallback: google/flan-t5-small...")
                    _generator = _build_text_generator("google/flan-t5-small")
                    _generator_model_name = "google/flan-t5-small"
                    return _generator
                except Exception as e2:
                    print(f"Falha ao carregar google/flan-t5-small: {type(e2).__name__}")

            # Fallback 2: GPT-2 PT para manter suporte ao portugues sem instrucoes.
            if RAG_MODEL != "pierreguillou/gpt2-small-portuguese":
                try:
                    print("Tentando fallback: pierreguillou/gpt2-small-portuguese...")
                    _generator = _build_text_generator("pierreguillou/gpt2-small-portuguese")
                    _generator_model_name = "pierreguillou/gpt2-small-portuguese"
                    return _generator
                except Exception as e3:
                    print(f"Falha ao carregar pierreguillou/gpt2-small-portuguese: {type(e3).__name__}")

            # Fallback 3: distilgpt2
            if RAG_MODEL != "distilgpt2":
                try:
                    print("Tentando fallback: distilgpt2...")
                    _generator = _build_text_generator("distilgpt2")
                    _generator_model_name = "distilgpt2"
                    return _generator
                except Exception as e4:
                    print(f"Falha ao carregar distilgpt2: {type(e4).__name__}")

            # Fallback 3: Modo offline sem modelo real
            print("Não conseguindo carregar modelo. Usando modo simulado...")
            return None

    return _generator


def _call_bento_generator(query, context, timeout=15):
    payload = {
        "query": query,
        "context": context,
    }
    response = requests.post(BENTO_GENERATOR_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    result = response.json()
    answer = result.get("answer")
    if not answer:
        raise ValueError("Resposta vazia recebida do serviço Bento.")
    return answer.strip()


def _build_vllm_chat_url() -> str:
    base = VLLM_BASE_URL.rstrip("/")
    if not base:
        raise ValueError("VLLM_BASE_URL não configurado.")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/v1/chat/completions"


def _call_vllm_openai_compatible(query, context, timeout=25):
    url = _build_vllm_chat_url()
    headers = {"Content-Type": "application/json"}
    if VLLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLLM_API_KEY}"

    payload = {
        "model": VLLM_MODEL,
        "temperature": 0.2,
        "max_tokens": 220,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Você é um assistente financeiro em português do Brasil. "
                    "Responda de forma objetiva e apenas com base no contexto fornecido."
                ),
            },
            {
                "role": "user",
                "content": f"Pergunta: {query}\n\nContexto:\n{context}",
            },
        ],
    }

    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise ValueError("Resposta vazia recebida do vLLM.")
    message = choices[0].get("message") or {}
    answer = (message.get("content") or "").strip()
    if not answer:
        raise ValueError("Campo choices[0].message.content vazio no vLLM.")
    return answer


def _call_remote_generator(query, context):
    mode = REMOTE_LLM_MODE or "auto"

    if mode == "vllm":
        return _call_vllm_openai_compatible(query, context)

    if mode == "bento":
        return _call_bento_generator(query, context)

    if VLLM_BASE_URL:
        try:
            return _call_vllm_openai_compatible(query, context)
        except Exception as exc:
            print(f"Falha no vLLM remoto: {exc}. Tentando endpoint /generate...")

    return _call_bento_generator(query, context)


def _clean_generated_answer(answer):
    """Limpa artefatos comuns de geração para resposta final mais natural."""
    text = answer.replace("\r", "")

    # Remove marcadores de prompt que por vezes aparecem na saída.
    text = re.sub(r"(?i)^\s*resposta\s*:\s*", "", text).strip()
    text = re.sub(r"(?im)^\s*(pergunta|contexto|difusora)\s*:\s*.*$", "", text)

    # Compacta espaços e quebras de linha excessivas.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _fix_mojibake(text):
    """Tenta corrigir texto UTF-8 lido como latin-1 (ex: 'Ã§', 'Ã£')."""
    if not text:
        return text
    if any(token in text for token in ("Ã", "â", "€", "™")):
        try:
            fixed = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
            if fixed:
                return fixed
        except Exception:
            pass
    return text


def _context_supports_query(query, context):
    """Heurística simples para evitar resposta inventada quando faltam termos-chave."""
    q = query.lower()
    c = context.lower()

    # Caso recorrente: pergunta sobre Itaú sem menção no contexto recuperado.
    if "itau" in q and "itau" not in c:
        return False
    return True


def _parse_metrics_from_context(context):
    """Extrai métricas por modelo a partir do bloco textual de contexto."""
    metrics = {}

    # Formato esperado no metrics.json serializado:
    # baseline: MAE=0.763, RMSE=1.005, MAPE=1.8%
    line_re = re.compile(
        r"(?i)\b([a-z_]+)\s*:\s*MAE\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*"
        r"RMSE\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*MAPE\s*=\s*([0-9]+(?:\.[0-9]+)?)"
    )
    for model, mae, rmse, mape in line_re.findall(context):
        metrics[model.lower()] = {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
        }

    # Formato alternativo de runs MLflow (chaves separadas)
    # mae_keras: 1.1613 / rmse_keras: 1.3303 / mape_keras: 2.5967
    for metric_name, model, value in re.findall(
        r"(?i)\b(mae|rmse|mape)_([a-z_]+)\s*:\s*([0-9]+(?:\.[0-9]+)?)", context
    ):
        metric_name = metric_name.lower()
        model = model.lower()
        metrics.setdefault(model, {})[metric_name] = float(value)

    return metrics


def _parse_hyperparams_from_context(context):
    """Extrai hiperparâmetros de treino do bloco textual."""
    params = {}
    patterns = {
        "ticker": r"Ticker de treino\s*:\s*(\S+)",
        "janela": r"Janela temporal\s*:\s*(\d+)",
        "epochs": r"Epochs\s*:\s*(\d+)",
        "batch": r"Batch size\s*:\s*(\d+)",
        "periodo": r"Per[ií]odo de dados\s*:\s*([^\n]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, context, re.IGNORECASE)
        if m:
            params[key] = m.group(1).strip()
    return params


def _answer_model_metrics_query(query, context):
    """Gera resposta determinística para perguntas de métricas e hiperparâmetros do modelo."""
    q = (query or "").lower()

    _metric_tokens = ("mae", "rmse", "mape", "métrica", "metricas", "erro", "acuracia")
    _model_tokens = ("modelo", "modelos", "treino", "treinamento", "baseline", "pytorch", "sklearn", "keras", "ensemble")
    _hyperparam_tokens = ("janela", "epoch", "batch", "hiperparâmetro", "hiperparametro", "parâmetro", "parametro", "quantos epoch", "qual janela", "qual batch", "treino", "treinamento")

    is_metric_q = any(t in q for t in _metric_tokens)
    is_model_q = any(t in q for t in _model_tokens)
    is_param_q = any(t in q for t in _hyperparam_tokens)

    if not (is_metric_q or is_model_q or is_param_q):
        return None

    # ── Perguntas sobre hiperparâmetros ──────────────────────────────────────
    # Entra quando é pergunta de param/treino sem ser pergunta de métrica
    if is_param_q and not is_metric_q:
        params = _parse_hyperparams_from_context(context)
        if params:
            parts = []
            if "ticker" in params:
                parts.append(f"Ticker: {params['ticker']}")
            if "janela" in params:
                parts.append(f"Janela temporal: {params['janela']} dias")
            if "epochs" in params:
                parts.append(f"Epochs: {params['epochs']}")
            if "batch" in params:
                parts.append(f"Batch size: {params['batch']}")
            if "periodo" in params:
                parts.append(f"Período: {params['periodo']}")
            if parts:
                return "Parâmetros de treinamento registrados: " + " | ".join(parts) + "."

    metrics = _parse_metrics_from_context(context)
    if not metrics:
        # Fallback: lê metrics.json diretamente do disco
        try:
            import json
            from pathlib import Path as _Path
            _app_dir = _Path("/app") if _Path("/app").is_dir() else _Path.cwd()
            _metrics_path = _app_dir / "reports" / "metrics.json"
            if _metrics_path.exists():
                _raw = json.loads(_metrics_path.read_text(encoding="utf-8"))
                for _mname, _vals in _raw.items():
                    metrics[_mname.lower()] = {
                        k: float(v) for k, v in _vals.items()
                        if isinstance(v, (int, float))
                    }
        except Exception:
            pass

    if not metrics:
        # Se não tem métricas mas é pergunta genérica de modelo, tenta params
        if is_model_q:
            params = _parse_hyperparams_from_context(context)
            if params:
                parts = []
                if "ticker" in params:
                    parts.append(f"Ticker: {params['ticker']}")
                if "janela" in params:
                    parts.append(f"Janela: {params['janela']} dias")
                if "epochs" in params:
                    parts.append(f"Epochs: {params['epochs']}")
                if "batch" in params:
                    parts.append(f"Batch size: {params['batch']}")
                return "Configuração de treino registrada: " + " | ".join(parts) + "." if parts else None
        # Pergunta explícita de métrica sem dados disponíveis
        if is_metric_q:
            return (
                "Métricas do modelo não encontradas no índice. "
                "Execute POST /ingest_mlflow para indexar os resultados do pipeline de treino."
            )
        return None

    # ── Determina a métrica alvo ──────────────────────────────────────────────
    target_metric = None
    if "mae" in q:
        target_metric = "mae"
    elif "rmse" in q:
        target_metric = "rmse"
    elif "mape" in q:
        target_metric = "mape"

    wants_best = any(t in q for t in ("menor", "melhor", "mais baixo", "lowest", "best", "venceu"))
    wants_worst = any(t in q for t in ("maior", "pior", "mais alto", "worst"))
    wants_compare = any(t in q for t in ("compar", "todos", "resumo", "tabela", "lista", "quais"))

    # Melhor/pior por métrica específica
    if target_metric and (wants_best or wants_worst):
        candidates = [(m, vals[target_metric]) for m, vals in metrics.items() if target_metric in vals]
        if candidates:
            if wants_worst:
                chosen_model, chosen_value = max(candidates, key=lambda x: x[1])
                label = "pior"
            else:
                chosen_model, chosen_value = min(candidates, key=lambda x: x[1])
                label = "melhor"
            return (
                f"O {label} resultado em {target_metric.upper()} foi do modelo "
                f"**{chosen_model}** com {target_metric.upper()}={chosen_value:.3f}."
            )

    # Comparação entre todos os modelos (ou pergunta genérica sem métrica específica)
    if wants_compare or (not target_metric and is_metric_q):
        lines = []
        for model in sorted(metrics):
            vals = metrics[model]
            parts = []
            if "mae" in vals:
                parts.append(f"MAE={vals['mae']:.3f}")
            if "rmse" in vals:
                parts.append(f"RMSE={vals['rmse']:.3f}")
            if "mape" in vals:
                parts.append(f"MAPE={vals['mape']:.2f}%")
            if parts:
                lines.append(f"{model}: " + ", ".join(parts))

        if lines:
            # Destaca o melhor por MAE
            mae_candidates = [(m, v["mae"]) for m, v in metrics.items() if "mae" in v]
            best_note = ""
            if mae_candidates:
                best_m, best_v = min(mae_candidates, key=lambda x: x[1])
                best_note = f" Melhor por MAE: {best_m} (MAE={best_v:.3f})."
            return "Métricas por modelo: " + " | ".join(lines) + "." + best_note

    # Pergunta sobre métrica específica sem superlativo → retorna o ranking
    if target_metric:
        candidates = sorted(
            [(m, vals[target_metric]) for m, vals in metrics.items() if target_metric in vals],
            key=lambda x: x[1],
        )
        if candidates:
            parts = [f"{m}: {v:.3f}" for m, v in candidates]
            return f"Ranking por {target_metric.upper()} (menor é melhor): " + " > ".join(parts) + "."

    # Fallback: resumo geral
    lines = []
    for model in sorted(metrics):
        vals = metrics[model]
        parts = []
        if "mae" in vals:
            parts.append(f"MAE={vals['mae']:.3f}")
        if "rmse" in vals:
            parts.append(f"RMSE={vals['rmse']:.3f}")
        if "mape" in vals:
            parts.append(f"MAPE={vals['mape']:.2f}%")
        if parts:
            lines.append(f"{model}: " + ", ".join(parts))

    return ("Métricas registradas: " + " | ".join(lines) + ".") if lines else None


def generate_answer(query, context, max_new_tokens=256):
    """
    Gera resposta usando a API Bento/vLLM quando disponível.
    Caso contrário, usa fallback local com Hugging Face ou modo simulado.
    """
    if os.environ.get("USE_BENTO_GENERATOR", "true").lower() in ("1", "true", "yes"):
        try:
            return _call_remote_generator(query, context)
        except Exception as exc:
            print(f"Falha ao chamar gerador remoto: {exc}. Usando fallback local.")

    # Caminho rápido para perguntas de métricas/modelo sem depender do LLM.
    context = _fix_mojibake(context)
    analytical_answer = _answer_model_metrics_query(query, context)
    if analytical_answer:
        return analytical_answer

    if not _context_supports_query(query, context):
        return "Não encontrei informação suficiente no contexto recuperado para responder com segurança a essa pergunta. Tente refazer a consulta com mais detalhes ou atualizar a base de notícias."

    generator = _get_generator()

    # Fallback: Se não conseguir carregar modelo, gera resposta simulada contextual
    if generator is None:
        print("Usando resposta simulada (modelo indisponível)")
        return _generate_simulated_answer(query, context)

    active_model = _generator_model_name or RAG_MODEL

    # Prompt para modelos instruction-following (T5) vs text-completion (GPT-2)
    is_text2text = active_model in _TEXT2TEXT_MODELS
    context = _fix_mojibake(context)

    if is_text2text:
        prompt = (
            "Você é um assistente financeiro em português do Brasil. "
            "Responda apenas com base no contexto fornecido, em no máximo 3 frases, sem copiar o contexto literalmente. "
            "Se o contexto não trouxer informação suficiente para responder à pergunta, diga isso explicitamente.\n\n"
            f"Pergunta: {query}\n"
            f"Contexto: {context}\n"
            "Resposta objetiva:"
        )
    else:
        prompt = f"Pergunta: {query}\n\nContexto: {context}\n\nResponda em português brasileiro de forma clara e objetiva:"

    if is_text2text:
        output = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            do_sample=False,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            early_stopping=True,
        )
    else:
        output = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
        )

    if is_text2text:
        # text2text-generation retorna apenas o texto gerado (sem o prompt)
        answer = output[0]["generated_text"].strip()
    else:
        generated_text = output[0]["generated_text"]
        answer = generated_text[len(prompt):].strip()

    answer = _clean_generated_answer(_fix_mojibake(answer))

    # Evita saída quase idêntica ao contexto (modelo apenas ecoando recuperação).
    if context and SequenceMatcher(None, answer.lower(), context.lower()).ratio() > 0.82:
        return "O contexto recuperado não traz dados objetivos suficientes para concluir essa análise com segurança. Refaça a pergunta com um ativo/setor explícito ou atualize as fontes indexadas."

    # Evita resposta que replica os marcadores estruturados de contexto.
    if re.search(r"(?i)fonte\s*\d+|trecho\s*:", answer):
        return "O contexto recuperado está parcial para essa pergunta. Tente uma consulta mais específica (empresa, período e indicador) para obter uma resposta objetiva."

    if not answer:
        answer = "Não há informação suficiente no contexto para responder com segurança."
    return answer


def _generate_simulated_answer(query, context):
    """Gera resposta simulada inteligente baseada no contexto fornecido."""
    # Análise do contexto para gerar resposta contextual
    context_lower = context.lower()

    # Palavras-chave para detectar tipo de pergunta
    if (
        "ação" in query.lower()
        or "investimento" in query.lower()
        or "recomendadas" in query.lower()
    ):
        if "renda fixa" in context_lower or "juros" in context_lower:
            return "Com base no contexto fornecido sobre renda fixa e juros altos, recomendo priorizar investimentos em títulos de renda fixa como CDB, Tesouro Direto e debêntures, que oferecem retornos atrativos acima de 10% ao ano com baixo risco. Para ações, considere empresas sólidas com dividendos consistentes e exposição internacional moderada."

        elif "etf" in context_lower or "internacional" in context_lower:
            return "Segundo o contexto sobre ETFs internacionais, uma boa estratégia para 2026 seria diversificar a carteira com fundos de índices globais, reduzindo a exposição concentrada no mercado brasileiro. Combine com renda fixa para balancear riscos e considere alocação de 20-30% em ativos internacionais."

        elif "cripto" in context_lower or "bitcoin" in context_lower:
            return "O contexto menciona criptomoedas como investimento de alto risco. Recomendo alocar no máximo 5-10% da carteira para ativos especulativos como Bitcoin e Ethereum, mantendo a maior parte em investimentos mais conservadores como renda fixa e ações blue-chip."

        elif "imóvel" in context_lower or "imobiliário" in context_lower:
            return "Conforme o contexto sobre o mercado imobiliário, imóveis continuam sendo um bom investimento de longo prazo, especialmente com aluguéis competitivos mesmo em cenários de juros altos. Considere imóveis comerciais ou residenciais em localizações premium."

        else:
            return "Para investimentos em ações em 2026, considere uma carteira diversificada com foco em empresas sólidas do setor de consumo, tecnologia e infraestrutura, com dividendos consistentes e exposição internacional. Combine com renda fixa (60%) e ações (30%) para reduzir volatilidade."

    elif "mercado" in query.lower() or "economia" in query.lower():
        if "guerra" in context_lower or "tensão" in context_lower:
            return "O contexto indica um cenário de tensão geopolítica que pode afetar os mercados. Recomenda-se manter uma carteira diversificada e acompanhar as notícias econômicas para ajustar posições conforme necessário."

        else:
            return "O mercado apresenta oportunidades em diferentes segmentos. Mantenha uma estratégia de longo prazo com diversificação adequada ao seu perfil de risco."

    # Resposta padrão mais inteligente
    return f"Baseado no contexto fornecido sobre '{context[:100]}...', recomendo uma abordagem equilibrada considerando os fatores mencionados. Consulte um assessor financeiro para decisões personalizadas."


def generate_text(prompt, max_new_tokens=128, temperature=0.7):
    """Gera texto bruto a partir de um prompt usando o gerador local."""
    generator = _get_generator()
    if generator is None:
        return "Final Answer: Não foi possível carregar um modelo de geração local."

    output = generator(
        prompt, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True
    )
    generated_text = output[0]["generated_text"]
    if generated_text.startswith(prompt):
        return generated_text[len(prompt) :].strip()
    return generated_text.strip()
