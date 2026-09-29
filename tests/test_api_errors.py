"""API input validation and error handling.

The agent is replaced with fakes, so these run offline and never call the LLM.
"""

import json

from fastapi.testclient import TestClient

from backend import main
from backend.models import streaming

client = TestClient(main.app)


def _events(response):
    return [
        json.loads(line[len("data: "):])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def test_empty_message_is_rejected():
    for message in ["", "   "]:
        assert client.post("/chat", json={"message": message}).status_code == 422
        assert (
            client.post("/chat/stream", json={"message": message}).status_code == 422
        )


def test_overlong_message_is_rejected():
    long_message = "a" * (main.MAX_MESSAGE_CHARS + 1)
    assert client.post("/chat", json={"message": long_message}).status_code == 422


def test_message_is_trimmed_before_the_agent_sees_it(monkeypatch):
    seen = []
    monkeypatch.setattr(
        main, "run", lambda m: seen.append(m) or {"answer": "hi", "intent": "chitchat"}
    )
    response = client.post("/chat", json={"message": "  hello  "})
    assert response.status_code == 200
    assert seen == ["hello"]


def test_chat_returns_503_when_the_agent_fails(monkeypatch):
    def boom(message):
        raise RuntimeError("provider down")

    monkeypatch.setattr(main, "run", boom)
    response = client.post("/chat", json={"message": "What is asthma?"})
    assert response.status_code == 503
    assert "provider down" not in response.text  # no internals leaked


def test_stream_ends_with_error_event_when_the_agent_fails(monkeypatch):
    def boom(message):
        raise RuntimeError("provider down")

    monkeypatch.setattr(streaming.intent_classifier, "classify", boom)
    response = client.post("/chat/stream", json={"message": "What is asthma?"})
    assert response.status_code == 200
    events = _events(response)
    assert events[-1]["type"] == "error"
    assert "provider down" not in response.text


def test_stream_canned_reply_when_no_symptoms_found(monkeypatch):
    monkeypatch.setattr(
        streaming.intent_classifier, "classify", lambda m: "symptom_report"
    )
    monkeypatch.setattr(streaming.symptom_extractor, "extract", lambda m: [])
    response = client.post("/chat/stream", json={"message": "I feel off"})
    types = [e["type"] for e in _events(response)]
    assert types == ["meta", "token", "done"]
