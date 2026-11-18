"""Unit tests for Phase 7: FastAPI Backend and Application Endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, load_model_and_tokenizer


@pytest.fixture(scope="module")
def client():
    # Explicitly load model and tokenizer before running tests
    load_model_and_tokenizer()
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model"] == "Mini-GPT"
    assert "device" in data


def test_model_info_endpoint(client: TestClient):
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["vocab_size"] > 0
    assert data["parameters"] > 0
    assert data["n_layer"] >= 1
    assert data["n_embd"] >= 16


def test_generate_endpoint_success(client: TestClient):
    payload = {
        "prompt": "ALICE: What is",
        "temperature": 0.7,
        "top_k": 10,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "max_tokens": 20,
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prompt"] == payload["prompt"]
    assert "completion" in data
    assert len(data["completion"]) > 0
    assert data["tokens_generated"] > 0
    assert data["generation_time_ms"] > 0


def test_chat_turn_endpoint(client: TestClient):
    payload = {
        "messages": [
            {"role": "user", "content": "What is self-attention?"}
        ],
        "temperature": 0.8,
        "top_k": 15,
        "max_tokens": 25,
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert isinstance(data["response"], str)
    assert data["tokens_generated"] > 0
    assert data["model"] == "Mini-GPT"


def test_clear_conversation_endpoint(client: TestClient):
    response = client.post("/api/clear")
    assert response.status_code == 200
    assert response.json()["status"] == "cleared"


def test_validation_errors(client: TestClient):
    # 1. Empty prompt error
    res_empty = client.post("/api/generate", json={"prompt": ""})
    assert res_empty.status_code == 422

    # 2. Invalid negative temperature
    res_neg_temp = client.post("/api/generate", json={"prompt": "Test", "temperature": -1.0})
    assert res_neg_temp.status_code == 422


def test_serve_frontend_root(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Mini GPT" in response.text

