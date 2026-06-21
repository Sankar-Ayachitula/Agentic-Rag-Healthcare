"""End-to-end tests of the FastAPI /chat endpoint.

Uses FastAPI's TestClient (real app, real handler, no running server) and hits
the real agent — so it calls the Groq LLM. Auto-skips if no key.
"""

import pytest
from fastapi.testclient import TestClient

from backend import config
from backend.main import app

client = TestClient(app)


def test_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.skipif(not config.GROQ_API_KEY, reason="needs GROQ_API_KEY")
def test_chat_medical_question():
    response = client.post("/chat", json={"message": "What is asthma?"})
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "medical_question"
    assert body["answer"]          # non-empty answer
    assert body["sources"]         # cited sources present


@pytest.mark.skipif(not config.GROQ_API_KEY, reason="needs GROQ_API_KEY")
def test_chat_symptom_report():
    response = client.post(
        "/chat",
        json={"message": "I have itching, skin rash and nodal skin eruptions"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "symptom_report"
    assert body["symptoms"]        # extracted at least one symptom
