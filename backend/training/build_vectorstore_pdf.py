"""Build the encyclopedia PDF vector store (Day 3).

Loads the medical encyclopedia, splits it into overlapping chunks, embeds each
chunk with a free local model, and stores the vectors in ChromaDB.

This is the slow one (~4,500 pages → ~17k chunks). Run from project root:
    python -m backend.training.build_vectorstore_pdf
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "data/encyclopedias/encyclopedia-of-medicine-vol-1-5-3rd-edition.pdf"
CHROMA_DIR = "chroma_db/encyclopedia"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 150    # characters shared between neighbours (preserves context)


def main():
    print("Loading PDF (4,505 pages — a few minutes)...", flush=True)
    pages = PyPDFLoader(PDF_PATH).load()   # one Document per page, with page no.
    print(f"Loaded {len(pages)} pages.", flush=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks.", flush=True)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    print("Embedding chunks (the slow part)...", flush=True)
    db = Chroma.from_documents(
        chunks, embedding=embeddings, persist_directory=CHROMA_DIR
    )
    print(f"Stored {len(chunks)} chunks -> {CHROMA_DIR}", flush=True)

    # Smoke test: retrieve real passages by meaning.
    query = "what are the symptoms of malaria?"
    print(f"\nTop matches for: '{query}'", flush=True)
    for hit in db.similarity_search(query, k=2):
        page = hit.metadata.get("page", "?")
        snippet = hit.page_content[:140].replace("\n", " ")
        print(f"  [page {page}] {snippet}...", flush=True)


if __name__ == "__main__":
    main()
