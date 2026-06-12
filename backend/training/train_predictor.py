"""Train the disease-prediction model (Day 2).

Pipeline:
  1. Load the symptom dataset + severity weights.
  2. Convert each patient row into a numeric feature vector
     (one column per known symptom, value = severity weight if present else 0).
  3. Train a RandomForest classifier.
  4. Evaluate accuracy on held-out data.
  5. Save the model + vocabulary so we can predict later.

Run from the project root:
    python -m backend.training.train_predictor
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Where things live (paths are relative to the project root)
DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("backend/models/artifacts")


def load_data():
    """Read the two CSVs we need."""
    df = pd.read_csv(DATA_DIR / "dataset.csv")
    severity = pd.read_csv(DATA_DIR / "Symptom-severity.csv")
    # The severity symptom names have stray spaces in a few places — clean them.
    severity["Symptom"] = severity["Symptom"].str.strip()
    return df, severity


def build_features(df, weight_of, vocab):
    """Turn each row of symptoms into a numeric vector.

    weight_of: dict like {"itching": 1, "high_fever": 7, ...}
    vocab:     the ordered list of every known symptom (our columns)

    For each patient row we start with all-zeros, then for every symptom
    they actually have we write its severity weight into that column.
    """
    rows = []
    for _, row in df.iterrows():
        vec = {symptom: 0 for symptom in vocab}          # all symptoms absent
        for cell in row[1:]:                              # skip the Disease column
            if pd.notna(cell):                            # ignore blank slots
                symptom = str(cell).strip()               # fix the stray spaces
                if symptom in vec:                        # known symptom?
                    vec[symptom] = weight_of[symptom]     # mark present, weighted
        rows.append(vec)
    # DataFrame with one column per symptom, in a fixed order
    return pd.DataFrame(rows)[vocab]


def main():
    df, severity = load_data()

    # A lookup table: symptom name -> severity weight
    weight_of = dict(zip(severity["Symptom"], severity["weight"]))
    # Our vocabulary = every symptom the model knows about (sorted for stability)
    vocab = sorted(weight_of.keys())
    print(f"Loaded {len(df)} rows, {df['Disease'].nunique()} diseases, "
          f"{len(vocab)} known symptoms.")

    # X = the numeric features, y = the disease label we want to predict
    X = build_features(df, weight_of, vocab)
    y = df["Disease"]

    # Hold out 20% of the data so we can honestly measure performance on
    # examples the model never saw. stratify=y keeps all 41 diseases balanced
    # across the split. random_state makes the split reproducible.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # A RandomForest = many decision trees voting together. Solid default.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Accuracy = fraction of test patients whose disease we predicted correctly.
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Test accuracy: {accuracy:.2%}")

    # Save the model AND the vocab/weights together — inference needs all three
    # to rebuild the same kind of feature vector from a user's symptoms.
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = ARTIFACTS_DIR / "predictor.joblib"
    joblib.dump(
        {"model": model, "vocab": vocab, "weight_of": weight_of},
        artifact_path,
    )
    print(f"Saved model -> {artifact_path}")


if __name__ == "__main__":
    main()
