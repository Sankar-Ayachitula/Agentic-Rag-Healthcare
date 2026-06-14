"""Build the symptom vector store (Day 3).

Turns each disease's description + precautions into one document, embeds them
with a free local model, and stores the vectors in ChromaDB so the RAG layer
can later retrieve them by meaning.

Run from the project root:
    python -m backend.training.build_vectorstore_symptom
"""

from pathlib import Path

import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db/symptoms"               # where the vectors are saved
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast, free

PRECAUTION_COLS = ["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]


def build_documents():
    """One document per disease, combining its description and precautions."""
    desc = pd.read_csv(DATA_DIR / "symptom_Description.csv")
    prec = pd.read_csv(DATA_DIR / "symptom_precaution.csv")
    merged = desc.merge(prec, on="Disease", how="left")

    docs = []
    for _, row in merged.iterrows():
        precautions = [str(row[c]) for c in PRECAUTION_COLS if pd.notna(row.get(c))]
        text = (
            f"Disease: {row['Disease']}.\n"
            f"Description: {row['Description']}\n"
            f"Precautions: {', '.join(precautions)}."
        )
        # metadata lets us trace a retrieved chunk back to its disease.
        docs.append(Document(page_content=text, metadata={"disease": row["Disease"]}))
    return docs


def main():
    docs = build_documents()
    print(f"Built {len(docs)} disease documents.")

    # The embedding model turns each document's text into a vector.
    # (First run downloads the model — a one-time ~90MB download.)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Embed all docs and persist them to disk in ChromaDB.
    db = Chroma.from_documents(docs, embedding=embeddings, persist_directory=CHROMA_DIR)
    print(f"Embedded & stored {len(docs)} documents -> {CHROMA_DIR}")

    # Smoke test: search by meaning, not keywords.
    query = "itchy red skin rash"
    print(f"\nTop matches for: '{query}'")
    for hit in db.similarity_search(query, k=3):
        print(f"  - {hit.metadata['disease']}")


if __name__ == "__main__":
    main()
