from transformers import pipeline

# 🔹 Inicializa o modelo de geração de texto
# Aqui usamos gpt2 como exemplo, pode trocar pelo modelo que já está servindo via BentoML/vLLM
generator = pipeline("text-generation", model="gpt2")

def generate_answer(query, context, max_length=200):
    """
    Gera resposta usando LLM com base na query e no contexto recuperado.
    
    Args:
        query (str): Pergunta do usuário.
        context (str): Texto recuperado pelo retriever (chunks concatenados).
        max_length (int): Tamanho máximo da resposta.
    
    Returns:
        str: Resposta gerada pelo modelo.
    """
    prompt = f"Pergunta: {query}\nContexto: {context}\nResposta:"
    output = generator(prompt, max_length=max_length, num_return_sequences=1)
    return output[0]["generated_text"]