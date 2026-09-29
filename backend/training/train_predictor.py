"""Train the disease-prediction model (Day 2).

Pipeline:
  1. Load the symptom dataset + severity weights.
  2. Convert each patient row into a numeric feature vector
     (one column per known symptom, value = severity weight if present else 0).
  3. Augment the training rows with random subsets of each disease's symptoms,
     because real users describe only a few symptoms, not the full list.
  4. Train a RandomForest classifier.
  5. Evaluate accuracy on held-out data.
  6. Save the model + vocabulary so we can predict later.

Run from the project root:
    python -m backend.training.train_predictor
"""

import random
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Where things live (paths are relative to the project root)
DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("backend/models/artifacts")

# Synthetic partial-symptom rows added per disease (see augment_with_subsets).
SUBSETS_PER_DISEASE = 200


def normalize_symptom(name):
    """Canonical symptom label.

    The two CSVs disagree on stray spaces (e.g. "dischromic _patches" vs
    "dischromic_patches", "foul_smell_of urine" vs "foul_smell_ofurine"), so
    drop every space. Without this those symptoms are silently ignored.
    """
    return str(name).replace(" ", "")


def load_data():
    """Read the two CSVs we need."""
    df = pd.read_csv(DATA_DIR / "dataset.csv")
    df["Disease"] = df["Disease"].str.strip()
    severity = pd.read_csv(DATA_DIR / "Symptom-severity.csv")
    # The symptom names have stray spaces in a few places — clean them.
    severity["Symptom"] = severity["Symptom"].map(normalize_symptom)
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
                symptom = normalize_symptom(cell)         # fix the stray spaces
                if symptom in vec:                        # known symptom?
                    vec[symptom] = weight_of[symptom]     # mark present, weighted
        rows.append(vec)
    # DataFrame with one column per symptom, in a fixed order
    return pd.DataFrame(rows)[vocab]


def augment_with_subsets(X, y, per_disease=SUBSETS_PER_DISEASE, seed=42):
    """Add rows that contain only a random subset of a disease's symptoms.

    The dataset's rows list most of a disease's symptoms, but a user typically
    mentions 2-4. Without these rows the forest ranks partial inputs poorly
    (e.g. fever + chills + headache + vomiting -> not Malaria).
    """
    rng = random.Random(seed)
    extra_X, extra_y = [], []
    for disease in sorted(y.unique()):
        # Every symptom column seen for this disease, with its weight.
        present = X[y == disease].max()
        symptoms = sorted(present[present > 0].index)
        for _ in range(per_disease):
            chosen = rng.sample(symptoms, rng.randint(2, len(symptoms)))
            extra_X.append({s: (present[s] if s in chosen else 0) for s in X.columns})
            extra_y.append(disease)
    X_aug = pd.concat([X, pd.DataFrame(extra_X)[list(X.columns)]], ignore_index=True)
    y_aug = pd.concat([y, pd.Series(extra_y)], ignore_index=True)
    return X_aug, y_aug


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

    # Augment only the training split, so the test set stays real rows.
    X_train, y_train = augment_with_subsets(X_train, y_train)

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
