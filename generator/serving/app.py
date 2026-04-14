import os

import bentoml
from bentoml.io import JSON
import requests

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://vllm:8000")
VLLM_MODEL = os.getenv("VLLM_MODEL", "facebook/opt-125m")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")

svc = bentoml.Service("acoes_generator")


@svc.api(input=JSON(), output=JSON())
def generate(input_json):
    query = input_json.get("query")
    context = input_json.get("context", "")
    prompt = f"Responda em português.\nPergunta: {query}\nContexto: {context}\nResposta:"

    headers = {}
    if VLLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLLM_API_KEY}"

    resp = requests.post(
        f"{VLLM_BASE_URL}/v1/completions",
        json={"model": VLLM_MODEL, "prompt": prompt, "max_tokens": 200},
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError(f"Resposta invalida do vLLM: {payload}")

    return {"answer": choices[0]["text"]}
