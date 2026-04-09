import bentoml
from bentoml.io import JSON
import requests

svc = bentoml.Service("acoes_generator")

@svc.api(input=JSON(), output=JSON())
def generate(input_json):
    query = input_json.get("query")
    context = input_json.get("context", "")
    prompt = f"Responda em português.\nPergunta: {query}\nContexto: {context}\nResposta:"

    resp = requests.post(
        "http://vllm:8000/v1/completions",
        json={
            "model": "facebook/opt-1.3b",
            "prompt": prompt,
            "max_tokens": 200
        }
    )
    return {"answer": resp.json()["choices"][0]["text"]}