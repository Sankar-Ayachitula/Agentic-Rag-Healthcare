"""Central configuration.

Loads settings from the environment (.env) so no secrets live in source.
Import these constants anywhere instead of calling os.getenv() all over.
"""

import os

from dotenv import load_dotenv

load_dotenv()  # reads key=value pairs from the .env file into the environment

# LLM provider: Groq (free, fast, OpenAI-compatible). Get a key at
# https://console.groq.com -> API Keys.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# Embeddings stay free and local — no provider needed.
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Where the vector stores were persisted on Day 3.
ENCYCLOPEDIA_DIR = "chroma_db/encyclopedia"
SYMPTOM_DIR = "chroma_db/symptoms"


def require_llm_key():
    """Fail loudly with a helpful message if the LLM key is missing."""
    if not GROQ_API_KEY or GROQ_API_KEY == "your-key-here":
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and add it to the .env file."
        )
