"""RAG chain (Day 4).

The "retrieve, then generate" pipeline:
  question -> find the most relevant stored text -> hand it to the LLM as
  context -> the LLM writes an answer grounded ONLY in that text.

This is what stops the model from making medical facts up.
"""

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

from backend import config
from backend.models.llm import get_llm

# Same embedding model we built the stores with — the query must be embedded
# the same way as the documents, or "nearest" means nothing.
_embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

# Open the two vector stores we built on Day 3 (read-only here).
_encyclopedia = Chroma(
    persist_directory=config.ENCYCLOPEDIA_DIR, embedding_function=_embeddings
)
_symptoms = Chroma(
    persist_directory=config.SYMPTOM_DIR, embedding_function=_embeddings
)

# The instructions that keep the LLM honest. {context} is filled with the
# retrieved text; {question} with the user's question.
_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a careful medical information assistant. "
            "Answer the question using ONLY the context below. "
            "If the context does not contain the answer, say you don't know — "
            "do not guess. Always remind the user to consult a real doctor. "
            "This is for education, not diagnosis.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

def retrieve(query, k=4, store="encyclopedia"):
    """Return the k most relevant chunks for a query."""
    db = _encyclopedia if store == "encyclopedia" else _symptoms
    return db.similarity_search(query, k=k)


def answer(question, k=4, store="encyclopedia"):
    """Retrieve context, then have the LLM answer grounded in it.

    Returns (answer_text, source_documents).
    """
    docs = retrieve(question, k=k, store=store)
    context = "\n\n".join(doc.page_content for doc in docs)
    messages = _PROMPT.format_messages(context=context, question=question)
    response = get_llm().invoke(messages)
    return response.content, docs


if __name__ == "__main__":
    text, sources = answer("What are the symptoms of malaria?")
    print("ANSWER:\n", text)
    print("\nSOURCES:")
    for doc in sources:
        print("  - page", doc.metadata.get("page", "?"))
