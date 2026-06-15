"""Disease-prediction model (Day 2) — inference side.

Loads the trained artifact and turns a list of symptom names into a
ranked list of probable diseases.
"""

from pathlib import Path

import joblib
import pandas as pd

ARTIFACT_PATH = Path("backend/models/artifacts/predictor.joblib")

# Load once at import time (cheap, and avoids re-reading the file every call).
_bundle = joblib.load(ARTIFACT_PATH)
_model = _bundle["model"]
_vocab = _bundle["vocab"]            # ordered list of known symptoms
_weight_of = _bundle["weight_of"]    # symptom name -> severity weight

# Public: the symptom vocabulary the model understands. The symptom extractor
# targets these exact labels when mapping free text.
VOCAB = _vocab


def _symptoms_to_features(symptoms):
    """Build the SAME kind of vector the model was trained on.

    Start all-zeros, then for each symptom the user has, write its
    severity weight into that column. Unknown symptoms are ignored.
    """
    vec = {symptom: 0 for symptom in _vocab}
    for s in symptoms:
        s = s.strip()
        if s in vec:
            vec[s] = _weight_of[s]
    # Return a one-row DataFrame with named columns in the exact training
    # order (_vocab) — this matches how the model was trained.
    return pd.DataFrame([vec])[_vocab]


def predict(symptoms, top_k=3):
    """Return the top_k most probable diseases for a list of symptoms.

    Example:
        predict(["high_fever", "chills", "headache"])
        -> [("Malaria", 0.41), ("Dengue", 0.22), ...]
    """
    features = _symptoms_to_features(symptoms)
    # predict_proba gives a probability for every one of the 41 diseases.
    probabilities = _model.predict_proba(features)[0]
    # Pair each disease name with its probability, sort high -> low.
    ranked = sorted(
        zip(_model.classes_, probabilities),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return ranked[:top_k]


if __name__ == "__main__":
    # A quick manual smoke test you can run directly.
    demo = ["high_fever", "chills", "headache", "vomiting", "nausea"]
    print(f"Symptoms: {demo}\n")
    for disease, prob in predict(demo):
        print(f"  {disease:<25} {prob:.1%}")
