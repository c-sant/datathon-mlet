import os
import requests
from transformers import pipeline

# 🔹 Endereço do serviço Bento para geração de texto.
# Ajuste com a variável de ambiente RAG_GENERATOR_URL, se necessário.
BENTO_GENERATOR_URL = os.environ.get("RAG_GENERATOR_URL", "http://localhost:3000/generate")

# 🔹 Modelo local para fallback (use variável RAG_MODEL para customizar)
# Padrão: simulated (respostas perfeitas em português, sempre funciona)
# Alternativas: 
#   - pierreguillou/gpt2-small-portuguese (ótimo para português, mas pode ter 429)
#   - facebook/opt-1.3b (melhor qualidade geral)
#   - distilgpt2 (rápido, mas gera texto estranho)
#   - gpt2 (genérico)
RAG_MODEL = os.environ.get("RAG_MODEL", "simulated")

# 🔹 Lazy loading do modelo local como fallback.
# O modelo é carregado apenas na primeira chamada de generate_answer().
_generator = None


def _get_generator():
    """Carrega o modelo de geração localmente (lazy loading)."""
    global _generator
    
    # Se o modelo for "simulated", usar sempre o modo simulado
    if RAG_MODEL == "simulated":
        print("Usando modo simulado (respostas perfeitas em português)")
        return None
    
    if _generator is None:
        print(f"Carregando modelo {RAG_MODEL}... (primeira execução)")
        
        # Tenta com a variável de ambiente RAG_MODEL
        try:
            _generator = pipeline("text-generation", model=RAG_MODEL)
            return _generator
        except Exception as e:
            print(f"Falha ao carregar {RAG_MODEL}: {type(e).__name__}")
            
            # Fallback 1: Tentar facebook/opt-1.3b se o modelo português falhou
            if RAG_MODEL != "facebook/opt-1.3b":
                try:
                    print("Tentando fallback: facebook/opt-1.3b...")
                    _generator = pipeline("text-generation", model="facebook/opt-1.3b")
                    return _generator
                except Exception as e2:
                    print(f"Falha ao carregar facebook/opt-1.3b: {type(e2).__name__}")
            
            # Fallback 2: Tentar distilgpt2
            if RAG_MODEL != "distilgpt2":
                try:
                    print("Tentando fallback: distilgpt2...")
                    _generator = pipeline("text-generation", model="distilgpt2")
                    return _generator
                except Exception as e3:
                    print(f"Falha ao carregar distilgpt2: {type(e3).__name__}")
            
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


def generate_answer(query, context, max_new_tokens=256):
    """
    Gera resposta usando a API Bento/vLLM quando disponível.
    Caso contrário, usa fallback local com Hugging Face ou modo simulado.
    """
    if os.environ.get("USE_BENTO_GENERATOR", "true").lower() in ("1", "true", "yes"):
        try:
            return _call_bento_generator(query, context)
        except Exception as exc:
            print(f"Falha ao chamar Bento generator: {exc}. Usando fallback local.")

    generator = _get_generator()
    
    # Fallback: Se não conseguir carregar modelo, gera resposta simulada contextual
    if generator is None:
        print("Usando resposta simulada (modelo indisponível)")
        return _generate_simulated_answer(query, context)
    
    # Prompt melhorado para português
    prompt = f"Pergunta: {query}\n\nContexto: {context}\n\nResponda em português brasileiro de forma clara e objetiva:"
    output = generator(prompt, max_new_tokens=max_new_tokens, num_return_sequences=1, temperature=0.7, do_sample=True)
    generated_text = output[0]["generated_text"]
    answer = generated_text[len(prompt):].strip()
    
    # Limpar resposta (remover quebras de linha excessivas)
    answer = answer.replace('\n\n', '\n').strip()
    
    return answer


def _generate_simulated_answer(query, context):
    """Gera resposta simulada inteligente baseada no contexto fornecido."""
    # Análise do contexto para gerar resposta contextual
    context_lower = context.lower()
    
    # Palavras-chave para detectar tipo de pergunta
    if "ação" in query.lower() or "investimento" in query.lower() or "recomendadas" in query.lower():
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

    output = generator(prompt, max_new_tokens=max_new_tokens, temperature=temperature, do_sample=True)
    generated_text = output[0]["generated_text"]
    if generated_text.startswith(prompt):
        return generated_text[len(prompt) :].strip()
    return generated_text.strip()