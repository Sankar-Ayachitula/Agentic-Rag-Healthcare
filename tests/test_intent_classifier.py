"""Integration tests for the intent classifier (calls the Groq LLM).

Auto-skips if no GROQ_API_KEY is configured.
"""

import pytest

from backend import config
from backend.models import intent_classifier

pytestmark = pytest.mark.skipif(
    not config.GROQ_API_KEY, reason="needs GROQ_API_KEY"
)


def test_symptom_report_intent():
    assert (
        intent_classifier.classify("I have a fever, chills and a headache")
        == "symptom_report"
    )


def test_medical_question_intent():
    assert (
        intent_classifier.classify("What is type 2 diabetes?")
        == "medical_question"
    )


def test_chitchat_intent():
    assert intent_classifier.classify("hello there!") == "chitchat"
