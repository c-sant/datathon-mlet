import os

import pytest
import requests


def test_generate():
    base_url = os.getenv("GENERATOR_SERVICE_URL")
    if not base_url:
        pytest.skip("Teste de integração requer a variável GENERATOR_SERVICE_URL configurada.")

    url = f"{base_url.rstrip('/')}/generate"
    payload = {
        "query": "Quais ações estão recomendadas para 2026?",
        "context": "Dados do retriever...",
    }

    response = requests.post(url, json=payload, timeout=10)
    response.raise_for_status()

    payload = response.json()
    assert "answer" in payload
    assert payload["answer"]


if __name__ == "__main__":
    test_generate()
