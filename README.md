# MedAssist-RAG

A hybrid **Agentic-RAG** healthcare decision-support chatbot, rebuilt from scratch.

It combines a symptom-based disease-prediction model with vector retrieval over
medical knowledge (symptom data + a medical encyclopedia) behind a LangGraph
agent, served via FastAPI with a React/Vite chat UI.

> ⚠️ For academic and learning purposes only. **Not for clinical use.**

---

## Architecture

```
user message
   │
   ▼
intent classifier ──► symptom extractor ──► disease predictor (RandomForest/XGBoost)
   │                                              │
   └────────────► RAG chain (Chroma + HF embeddings + OpenAI) 
                                                  │
                          orchestrator (LangGraph) merges ──► grounded answer
```

## Stack
- **Backend:** Python, FastAPI, LangGraph, LangChain
- **Prediction:** scikit-learn / XGBoost
- **RAG:** ChromaDB + `sentence-transformers` embeddings; OpenAI for generation
- **Frontend:** React + Vite

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your OPENAI_API_KEY
```

See `data/README.md` for obtaining the datasets and encyclopedia PDF.

## Build steps (first run)
```bash
python -m backend.training.train_predictor
python -m backend.training.build_vectorstore_symptom
python -m backend.training.build_vectorstore_pdf
```

## Run
```bash
uvicorn backend.main:app --reload      # backend  -> http://127.0.0.1:8000
cd frontend && npm install && npm run start   # frontend -> http://localhost:3000
```

---

## Build log
- **Day 1** — scaffold, env, data exploration ✅ *(in progress)*
- Day 2 — disease-prediction model
- Day 3 — vector stores / embeddings
- Day 4 — RAG chain + intent + symptom extraction
- Day 5 — LangGraph orchestrator
- Day 6 — FastAPI + React wiring
- Day 7 — tests, eval, polish
