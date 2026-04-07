import bentoml
from bentoml.io import JSON
import requests

svc = bentoml.Service(
    "acoes_generator",
    description="Serviço de geração de texto em português usando vLLM",
)

@svc.api(
    input=JSON(
        example={
            "query": "Quais ações estão recomendadas para 2026?",
            "context": "Dados do retriever..."
        }
    ),
    output=JSON(
        example={
            "answer": "As ações recomendadas incluem diversificação em setores de tecnologia e energia renovável."
        }
    ),
)
def generate(input_json):
    query = input_json.get("query")
    context = input_json.get("context", "")
    prompt = f"Responda em português.\nPergunta: {query}\nContexto: {context}\nResposta:"

    resp = requests.post(
        "http://vllm:8000/v1/completions",
        json={
            "model": "pierreguillou/gpt2-small-portuguese",
            "prompt": prompt,
            "max_tokens": 200
        }
    )
    return {"answer": resp.json()["choices"][0]["text"]}