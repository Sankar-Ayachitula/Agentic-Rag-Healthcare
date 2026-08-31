"""Streaming answer generation (Day 6+).

Runs the same routing as the orchestrator (classify -> predict/retrieve), then
streams ONLY the final LLM answer token-by-token as Server-Sent Events (SSE).

SSE protocol (one JSON object per `data:` line):
  {"type": "meta",  "intent": ..., "disease": ..., "symptoms": [...], "sources": [pages]}
  {"type": "token", "text": "..."}      # many of these, in order
  {"type": "done"}
"""

import json

from langchain_core.prompts import ChatPromptTemplate

from backend.models import intent_classifier, predictor, rag_chain, symptom_extractor
from backend.models.llm import get_llm

_SYMPTOM_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful medical assistant. The user described symptoms, and "
            "a model predicted the most likely condition is '{disease}'. Using ONLY "
            "the context below, briefly explain that condition and its precautions. "
            "Make clear this is not a diagnosis and they should consult a real "
            "doctor. Education only.\n\nContext:\n{context}",
        ),
        ("human", "My symptoms: {symptoms}"),
    ]
)

_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful medical information assistant. Answer the question "
            "using ONLY the context below. If the context does not contain the "
            "answer, say you don't know, do not guess. Always remind the user to "
            "consult a real doctor. This is for education, not diagnosis.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

_CHITCHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a friendly medical-assistant chatbot. Reply briefly, and "
            "invite the user to describe symptoms or ask a health question.",
        ),
        ("human", "{message}"),
    ]
)


def _sse(obj):
    """Format a dict as one SSE event."""
    return f"data: {json.dumps(obj)}\n\n"


def _pages(docs):
    """Extract page numbers from retrieved docs (encyclopedia has them)."""
    return [d.metadata["page"] for d in docs if "page" in d.metadata]


def stream_events(message):
    """Yield SSE strings: one meta event, then token events, then done."""
    intent = intent_classifier.classify(message)
    disease = None
    symptoms = []
    pages = []
    messages = None
    canned = None  # a fixed reply that skips the LLM (e.g. no symptoms found)

    if intent == "symptom_report":
        symptoms = symptom_extractor.extract(message)
        if not symptoms:
            canned = (
                "I couldn't pick out specific symptoms. Could you describe what "
                "you're feeling in a bit more detail?"
            )
        else:
            disease = predictor.predict(symptoms, top_k=1)[0][0]
            docs = rag_chain.retrieve(disease, k=2, store="symptoms")
            context = "\n\n".join(d.page_content for d in docs)
            messages = _SYMPTOM_PROMPT.format_messages(
                disease=disease, context=context, symptoms=", ".join(symptoms)
            )
    elif intent == "medical_question":
        docs = rag_chain.retrieve(message, k=4, store="encyclopedia")
        context = "\n\n".join(d.page_content for d in docs)
        pages = _pages(docs)
        messages = _QUESTION_PROMPT.format_messages(context=context, question=message)
    else:
        messages = _CHITCHAT_PROMPT.format_messages(message=message)

    # 1. metadata first, so the client can render the disease chip / sources.
    yield _sse(
        {
            "type": "meta",
            "intent": intent,
            "disease": disease,
            "symptoms": symptoms,
            "sources": pages,
        }
    )

    # 2. the answer, token by token.
    if canned is not None:
        yield _sse({"type": "token", "text": canned})
    else:
        for chunk in get_llm().stream(messages):
            if chunk.content:
                yield _sse({"type": "token", "text": chunk.content})

    # 3. end of stream.
    yield _sse({"type": "done"})
