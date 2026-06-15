# Agentic RAG Healthcare

A hybrid **Agentic-RAG** healthcare decision-support chatbot, rebuilt from scratch.

A LangGraph **agent** reads each message, decides what to do, and routes it:
a symptom report runs a disease-prediction model and explains the result; a
general medical question is answered with retrieval over a medical encyclopedia;
small talk just gets a reply. Answers are **grounded** in retrieved text and
**cited**, so the LLM can't make medical facts up.

> ⚠️ For academic and learning purposes only. **Not for clinical use.**

---

## Architecture

```
user message
   │
   ▼
intent classifier ───┬── symptom_report  ─► extract symptoms ─► disease classifier ─┐
   (LangGraph agent)  │                                                              │
                      ├── medical_question ─────────────────────────────────────┐   │
                      └── chitchat ─► plain reply                                │   │
                                                                                 ▼   ▼
                                              RAG: retrieve from vector stores ─► LLM ─► grounded, cited answer
                                              (symptom store + encyclopedia)
```

- The **classifier** predicts the disease (it never retrieves).
- The **vector stores** supply grounded knowledge (they never predict).
- The **agent** decides which path/tools each message needs.

## Stack
- **Orchestration:** LangGraph (agentic routing) + LangChain
- **Prediction:** scikit-learn `RandomForestClassifier`
- **RAG:** ChromaDB vector stores; `sentence-transformers` embeddings (free, local)
- **LLM:** Groq (`llama-3.3-70b-versatile`) — free, fast, OpenAI-compatible
- **API:** FastAPI (Day 6)
- **Client:** native Android app — Kotlin + Jetpack Compose (Day 6)

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your free GROQ_API_KEY (console.groq.com)
```
Embeddings run locally, so only the LLM needs a key. See `data/README.md` for
obtaining the datasets and encyclopedia PDF.

## Build steps (first run)
```bash
python -m backend.training.train_predictor            # train + save the classifier
python -m backend.training.build_vectorstore_symptom  # 41 disease cards -> Chroma
python -m backend.training.build_vectorstore_pdf      # encyclopedia -> Chroma (slow)
```

## Try it (CLI)
```bash
python -m backend.models.orchestrator   # runs sample messages through the agent
```

## Data sources
All data is **public**; none is real patient data.
- **Symptom → disease + severity / descriptions / precautions:** the *Disease
  Symptom Prediction* dataset (Kaggle) — 41 diseases, curated/synthetic.
- **Secondary symptom set:** Mendeley Data (optional).
- **Knowledge base for RAG:** *Gale Encyclopedia of Medicine, 3rd ed.*

The Kaggle data is clean and balanced by design — great for demonstrating the
pipeline, not for clinical claims.

---

## Build log
- **Day 1** — scaffold, env, data exploration ✅
- **Day 2** — disease-prediction model ✅
- **Day 3** — vector stores / embeddings ✅
- **Day 4** — RAG chain + intent classifier + symptom extractor ✅
- **Day 5** — LangGraph orchestrator (full pipeline) ✅
- **Day 6** — FastAPI endpoint + Android client ⬜
- **Day 7** — tests, eval, polish ⬜
