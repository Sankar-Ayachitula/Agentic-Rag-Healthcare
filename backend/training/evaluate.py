"""Evaluation metrics (Day 7).

Deterministic, reproducible, defensible evals:

  1. Classifier: 5-fold cross-validated accuracy / precision / recall / F1.
  2. Encyclopedia retrieval recall@k: query "What is {disease}?" and count a hit
     if the target disease term appears in a retrieved chunk. This is the RAG
     path the app uses for medical questions (the headline retrieval metric).
  3. Symptom-store retrieval recall@k: query by raw symptoms and check whether
     the correct disease card is retrieved (a harder cross-modal match).

Run from the project root:
    python -m backend.training.evaluate
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from backend.training.train_predictor import build_features, load_data

KS = (1, 3, 5)


def evaluate_classifier():
    df, severity = load_data()
    weight_of = dict(zip(severity["Symptom"], severity["weight"]))
    vocab = sorted(weight_of.keys())
    X = build_features(df, weight_of, vocab)
    y = df["Disease"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = cross_val_predict(model, X, y, cv=skf)

    print("=== CLASSIFIER (5-fold cross-validation) ===")
    print(f"accuracy        : {accuracy_score(y, preds):.3f}")
    print(f"precision(macro): {precision_score(y, preds, average='macro', zero_division=0):.3f}")
    print(f"recall(macro)   : {recall_score(y, preds, average='macro', zero_division=0):.3f}")
    print(f"f1(macro)       : {f1_score(y, preds, average='macro'):.3f}")
    print("note: ~1.0 reflects the clean synthetic dataset, not clinical readiness.")


def _base_term(name):
    cleaned = name.lower().replace("(", " ").replace(")", " ").strip()
    tokens = cleaned.split()
    return tokens[0] if tokens else name.lower()


def evaluate_encyclopedia_retrieval():
    from backend.models import rag_chain

    diseases = [
        d.strip()
        for d in pd.read_csv("data/symptom_Description.csv")["Disease"].tolist()
    ]
    hits = {k: 0 for k in KS}
    for d in diseases:
        docs = rag_chain.retrieve(f"What is {d}?", k=max(KS), store="encyclopedia")
        texts = [doc.page_content.lower() for doc in docs]
        term = _base_term(d)
        for k in KS:
            if any(term in t for t in texts[:k]):
                hits[k] += 1

    print("\n=== ENCYCLOPEDIA recall@k (query 'What is {disease}?') ===")
    print(f"queries: {len(diseases)} | hit = target disease appears in a retrieved chunk")
    for k in KS:
        print(f"recall@{k}: {hits[k] / len(diseases):.3f}")


def evaluate_symptom_retrieval():
    from backend.models import rag_chain

    data = pd.read_csv("data/dataset.csv")
    reps = data.groupby("Disease").first().reset_index()
    sym_cols = [c for c in data.columns if c != "Disease"]

    hits = {k: 0 for k in KS}
    total = 0
    for _, row in reps.iterrows():
        true = row["Disease"].strip().lower()
        symptoms = [
            str(row[c]).strip().replace("_", " ")
            for c in sym_cols
            if pd.notna(row[c])
        ]
        if not symptoms:
            continue
        docs = rag_chain.retrieve("I have " + ", ".join(symptoms), k=max(KS), store="symptoms")
        retrieved = [doc.metadata["disease"].strip().lower() for doc in docs]
        total += 1
        for k in KS:
            if true in retrieved[:k]:
                hits[k] += 1

    print("\n=== SYMPTOM-STORE recall@k (query = raw symptoms; harder) ===")
    print(f"queries: {total}")
    for k in KS:
        print(f"recall@{k}: {hits[k] / total:.3f}")


if __name__ == "__main__":
    evaluate_classifier()
    evaluate_encyclopedia_retrieval()
    evaluate_symptom_retrieval()
