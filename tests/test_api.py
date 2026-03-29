from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "ok"
    assert "model_version" in payload


def test_predict_returns_expected_schema():
    response = client.post("/predict", json={"text": "hello world"})
    assert response.status_code == 200

    payload = response.json()
    assert "label" in payload
    assert "score" in payload
    assert "model_version" in payload
