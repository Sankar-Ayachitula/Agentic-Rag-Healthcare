"""Symptom extractor (Day 4).

Maps a user's free-text message onto the classifier's fixed symptom
vocabulary. This is the bridge: "I feel hot and shivery" -> ["high_fever",
"chills"], using the exact labels the disease classifier was trained on.
"""

import json

from langchain_core.prompts import ChatPromptTemplate

from backend.models import predictor
from backend.models.llm import get_llm

# The exact labels the classifier understands (from the trained artifact).
_VOCAB = predictor.VOCAB

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You extract medical symptoms from a message. You are given a fixed "
            "list of valid symptom labels. Return ONLY the labels from that list "
            "that the message describes, as a JSON array of strings. Map everyday "
            "wording to the closest label (e.g. 'throwing up' -> 'vomiting'). "
            "If none apply, return []. Output JSON only, no other text.\n\n"
            "Valid symptom labels:\n{vocab}",
        ),
        ("human", "{message}"),
    ]
)


def extract(message):
    """Return a list of known symptom labels found in the message."""
    messages = _PROMPT.format_messages(vocab=", ".join(_VOCAB), message=message)
    raw = get_llm().invoke(messages).content.strip()

    # The model should return JSON; parse it, and guard against stray text.
    try:
        labels = json.loads(raw)
    except json.JSONDecodeError:
        # Sometimes models wrap JSON in ```code fences``` — strip and retry.
        cleaned = raw.strip("`").replace("json", "", 1).strip()
        labels = json.loads(cleaned)

    # Keep only labels that really exist in our vocabulary (drop hallucinations).
    valid = set(_VOCAB)
    return [label for label in labels if label in valid]


if __name__ == "__main__":
    msg = "I've been feeling really hot and shivery with a pounding head and I keep throwing up"
    print(f"Message: {msg}\n")
    print("Extracted symptoms:", extract(msg))
