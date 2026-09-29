"""FastAPI application (Day 6).

Exposes the agent over HTTP so any client (the Android app, a browser, curl)
can talk to it. The orchestrator does all the work; this is just the doorway.

Run from the project root:
    uvicorn backend.main:app --reload
"""

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from backend.models.orchestrator import run
from backend.models.streaming import stream_events

logger = logging.getLogger(__name__)

# Longest message we accept; keeps a single request from blowing up the LLM prompt.
MAX_MESSAGE_CHARS = 2000

app = FastAPI(title="Agentic RAG Healthcare", version="1.0")

# Allow any origin during development so the Android emulator / a browser can
# call the API freely. Tighten this to known origins before any real deploy.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """What the client sends."""
    message: str = Field(..., max_length=MAX_MESSAGE_CHARS)

    @field_validator("message")
    @classmethod
    def not_blank(cls, value):
        """Reject empty / whitespace-only messages (FastAPI returns 422)."""
        value = value.strip()
        if not value:
            raise ValueError("message must not be empty")
        return value


class ChatResponse(BaseModel):
    """What we send back. Optional fields are only set on the symptom path."""
    answer: str
    intent: Optional[str] = None
    disease: Optional[str] = None
    symptoms: Optional[List[str]] = None
    sources: Optional[list] = None


@app.get("/")
def health():
    """Simple liveness check."""
    return {"status": "ok", "service": "agentic-rag-healthcare"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Run a user message through the agent and return its answer (all at once)."""
    try:
        result = run(request.message)
    except Exception:
        # Usually the LLM provider (bad key, rate limit, outage). Log the details
        # server-side; give the client a clean error instead of a stack trace.
        logger.exception("chat failed")
        raise HTTPException(
            status_code=503, detail="The assistant is temporarily unavailable."
        )
    return ChatResponse(
        answer=result.get("answer", ""),
        intent=result.get("intent"),
        disease=result.get("disease"),
        symptoms=result.get("symptoms"),
        sources=result.get("sources"),
    )


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """Same routing as /chat, but streams the answer token-by-token as SSE."""
    return StreamingResponse(
        stream_events(request.message), media_type="text/event-stream"
    )
