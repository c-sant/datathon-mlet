import requests

def test_generate():
    url = "http://localhost:3000/generate"
    payload = {
        "query": "Quais ações estão recomendadas para 2026?",
        "context": "Dados do retriever..."
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("✅ Resposta recebida:")
        print(response.json())
    except requests.exceptions.RequestException as e:
        print("❌ Erro na requisição:", e)

if __name__ == "__main__":
    test_generate()