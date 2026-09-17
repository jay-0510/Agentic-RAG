from fastapi.testclient import TestClient
import pytest

from app.core.config import get_settings
from app.main import app, get_graph

client = TestClient(app)


@pytest.fixture(autouse=True)
def use_local_backends(monkeypatch):
    """Tests must not call configured cloud services or consume model credits."""
    monkeypatch.setenv("LLM_BACKEND", "local")
    monkeypatch.setenv("RETRIEVER_BACKEND", "local")
    get_settings.cache_clear()
    get_graph.cache_clear()
    yield
    get_settings.cache_clear()
    get_graph.cache_clear()


def test_health_check():
    assert client.get("/health").json() == {"status": "ok"}


def test_sports_question_retrieves_context():
    response = client.post("/v1/ask", json={"question": "Who is Virat Kohli?"})
    body = response.json()
    assert response.status_code == 200
    assert body["used_retrieval"] is True
    assert body["citations"]
    assert "virat kohli" in body["answer"].lower()


def test_greeting_skips_retrieval():
    response = client.post("/v1/ask", json={"question": "hello"})
    assert response.status_code == 200
    assert response.json()["used_retrieval"] is False
