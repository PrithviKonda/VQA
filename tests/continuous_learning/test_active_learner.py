import pytest
from src.continuous_learning.active_learner import ActiveLearner

def test_entropy():
    al = ActiveLearner()
    probs = [0.5, 0.5]
    assert abs(al.entropy(probs) - 0.6931) < 0.01

def test_entropy_empty():
    al = ActiveLearner()
    assert al.entropy([]) == 0.0

def test_margin():
    al = ActiveLearner()
    probs = [0.7, 0.2, 0.1]
    assert abs(al.uncertainty_score(probs, method="margin") - (1.0 - (0.7 - 0.2))) < 0.01

def test_select_high_uncertainty_samples():
    al = ActiveLearner()
    samples = [
        {"probs": [0.7, 0.2, 0.1], "id": 1},
        {"probs": [0.34, 0.33, 0.33], "id": 2},
        {"probs": [0.9, 0.05, 0.05], "id": 3}
    ]
    selected = al.select_high_uncertainty_samples(samples, k=1)
    assert selected[0]["id"] == 2

def test_select_high_uncertainty_samples_empty():
    al = ActiveLearner()
    samples = []
    selected = al.select_high_uncertainty_samples(samples, k=1)
    assert selected == []
