"""FastAPI application (Day 6).

Exposes the agent over HTTP so any client (the Android app, a browser, curl)
can talk to it. The orchestrator does all the work; this is just the doorway.

Run from the project root:
    uvicorn backend.main:app --reload
"""

from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.models.orchestrator import run
from backend.models.streaming import stream_events

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
    message: str


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
    result = run(request.message)
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
