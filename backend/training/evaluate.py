"""Evaluation metrics (Day 7).

Two deterministic, defensible evals:
  1. Classifier: 5-fold cross-validated accuracy / precision / recall / F1.
  2. Retrieval: recall@k on the symptom store, querying by a disease's symptoms
     and checking whether its card is retrieved.

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


def evaluate_retrieval(ks=(1, 3, 5)):
    # Import here so the classifier eval can run without loading embeddings.
    from backend.models import rag_chain

    data = pd.read_csv("data/dataset.csv")
    reps = data.groupby("Disease").first().reset_index()
    sym_cols = [c for c in data.columns if c != "Disease"]

    hits = {k: 0 for k in ks}
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
        query = "I have " + ", ".join(symptoms)
        docs = rag_chain.retrieve(query, k=max(ks), store="symptoms")
        retrieved = [d.metadata["disease"].strip().lower() for d in docs]
        total += 1
        for k in ks:
            if true in retrieved[:k]:
                hits[k] += 1

    print("\n=== RETRIEVAL recall@k (symptom store, query = symptoms) ===")
    print(f"diseases evaluated: {total}")
    for k in ks:
        print(f"recall@{k}: {hits[k] / total:.3f}")


if __name__ == "__main__":
    evaluate_classifier()
    evaluate_retrieval()
