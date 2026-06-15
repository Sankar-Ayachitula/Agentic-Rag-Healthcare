"""Shared Groq chat model.

One lazily-created instance reused across the RAG chain, intent classifier,
and symptom extractor — avoids spinning up a separate client per module.
"""

from langchain_groq import ChatGroq

from backend import config

_llm = None


def get_llm():
    """Return the shared ChatGroq instance (created on first use)."""
    global _llm
    if _llm is None:
        config.require_llm_key()
        # temperature=0 -> deterministic, factual output (no creativity).
        _llm = ChatGroq(
            model=config.LLM_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=0,
        )
    return _llm
