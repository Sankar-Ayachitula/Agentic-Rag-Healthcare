"""Unit tests for parsing the symptom extractor's LLM output.

No LLM call: these feed raw model replies straight into the parser.
"""

from backend.models import symptom_extractor


def test_parses_plain_json_array():
    assert symptom_extractor._parse_labels('["chills", "headache"]') == [
        "chills",
        "headache",
    ]


def test_parses_code_fenced_json():
    raw = '```json\n["vomiting"]\n```'
    assert symptom_extractor._parse_labels(raw) == ["vomiting"]


def test_parses_array_wrapped_in_prose():
    raw = 'Here are the symptoms: ["high_fever"]. Hope that helps!'
    assert symptom_extractor._parse_labels(raw) == ["high_fever"]


def test_garbage_returns_empty_list():
    assert symptom_extractor._parse_labels("I can't help with that.") == []
    assert symptom_extractor._parse_labels("[not json]") == []
    assert symptom_extractor._parse_labels('{"a": 1}') == []


def test_extract_drops_unknown_and_duplicate_labels(monkeypatch):
    class FakeLLM:
        def invoke(self, messages):
            class Reply:
                content = '["chills", "made_up_symptom", "chills"]'
            return Reply()

    monkeypatch.setattr(symptom_extractor, "get_llm", lambda: FakeLLM())
    assert symptom_extractor.extract("I'm shivering") == ["chills"]
