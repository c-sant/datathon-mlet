from fastapi.testclient import TestClient
from src.rag.api import app

client = TestClient(app)

resp = client.get('/query', params={'q': 'Quais ações estão recomendadas para 2026?', 'top_k': 3})
print('/query', resp.status_code)
print(resp.json())

resp2 = client.post('/ingest', json={'docs': [{'id': 'test1', 'title': 'Teste', 'text': 'Mercado financeiro e ações.'}], 'overwrite': True})
print('/ingest', resp2.status_code)
print(resp2.json())
