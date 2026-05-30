"""FastAPI application entrypoint (wired on Day 6).

Will expose a POST /chat endpoint that passes a user message to the
orchestrator and returns the grounded answer, plus CORS for the Vite frontend.
Run with: uvicorn backend.main:app --reload
"""
