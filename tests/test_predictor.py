"""Tests for the disease predictor.

Fast and deterministic: no LLM, no network. These always run.
"""

from backend.models import predictor


def test_predict_returns_ranked_tuples():
    result = predictor.predict(["itching", "skin_rash"], top_k=3)
    assert len(result) == 3
    for disease, prob in result:
        assert isinstance(disease, str)
        assert 0.0 <= prob <= 1.0
    # Probabilities must come back sorted high to low.
    probs = [p for _, p in result]
    assert probs == sorted(probs, reverse=True)


def test_known_skin_symptoms_predict_fungal_infection():
    top = predictor.predict(
        ["itching", "skin_rash", "nodal_skin_eruptions"], top_k=1
    )[0][0]
    assert top == "Fungal infection"


def test_unknown_symptoms_do_not_crash():
    # Made-up symptoms should be ignored, not raise.
    result = predictor.predict(["not_a_real_symptom"], top_k=3)
    assert len(result) == 3


def test_partial_symptoms_rank_the_matching_disease_first():
    # Users mention only a few symptoms; all four of these are Malaria's.
    top = predictor.predict(
        ["high_fever", "chills", "headache", "vomiting"], top_k=1
    )[0][0]
    assert top == "Malaria"


def test_vocab_has_no_stray_spaces():
    # The CSVs disagree on spaces ("dischromic _patches"); names are normalized.
    assert all(" " not in symptom for symptom in predictor.VOCAB)
    assert "dischromic_patches" in predictor.VOCAB


def test_input_symptom_names_are_normalized():
    assert predictor.predict(["dischromic _patches"], top_k=1) == predictor.predict(
        ["dischromic_patches"], top_k=1
    )
