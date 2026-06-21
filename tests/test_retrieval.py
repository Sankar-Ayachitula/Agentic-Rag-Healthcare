"""Tests for vector-store retrieval.

Uses the local embedding model + Chroma stores (no LLM). Deterministic.
"""

from backend.models import rag_chain


def test_encyclopedia_retrieves_relevant_chunk():
    docs = rag_chain.retrieve("symptoms of malaria", k=4, store="encyclopedia")
    assert len(docs) == 4
    combined = " ".join(doc.page_content.lower() for doc in docs)
    assert "malaria" in combined


def test_symptom_store_returns_matching_disease():
    docs = rag_chain.retrieve("Malaria", k=1, store="symptoms")
    assert docs[0].metadata["disease"] == "Malaria"
