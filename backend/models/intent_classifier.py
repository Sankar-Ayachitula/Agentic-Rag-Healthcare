"""Intent classifier (Day 4).

Decides what kind of message the user sent, so the orchestrator (Day 5) knows
which path to take:
  - "symptom_report"   -> run the symptom extractor + disease classifier
  - "medical_question" -> answer with RAG only
  - "chitchat"         -> just reply, no prediction or retrieval
"""

from langchain_core.prompts import ChatPromptTemplate

from backend.models.llm import get_llm

VALID_INTENTS = {"symptom_report", "medical_question", "chitchat"}

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Classify the user's message into exactly one of these labels:\n"
            "- symptom_report: they describe symptoms they are experiencing\n"
            "- medical_question: a general medical/health question\n"
            "- chitchat: greetings or anything not medical\n"
            "Reply with ONLY the label, nothing else.",
        ),
        ("human", "{message}"),
    ]
)


def classify(message):
    """Return one of VALID_INTENTS for the given message."""
    response = get_llm().invoke(_PROMPT.format_messages(message=message))
    label = response.content.strip().lower()
    # Be defensive: if the model adds stray text, fall back to a safe default.
    return label if label in VALID_INTENTS else "medical_question"


if __name__ == "__main__":
    tests = [
        "I have a fever, chills, and a headache",
        "What is type 2 diabetes?",
        "hey there!",
    ]
    for msg in tests:
        print(f"{classify(msg):<18} <- {msg}")
