# src/continuous_learning/active_learner.py
"""
Logic for selecting data for labeling (uncertainty sampling).
"""
from typing import List, Dict, Any
import numpy as np

class ActiveLearner:
    """
    Implements uncertainty-based sample selection for active learning.
    """
    def __init__(self):
        pass

    @staticmethod
    def entropy(probabilities: List[float]) -> float:
        """
        Calculate the entropy of a probability distribution.
        """
        probs = np.array(probabilities)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))

    def uncertainty_score(self, output_probs: List[float], method: str = "entropy") -> float:
        """
        Compute uncertainty score for a model's output probabilities.
        Supported methods: 'entropy', 'margin'.
        """
        if method == "entropy":
            return self.entropy(output_probs)
        elif method == "margin":
            sorted_probs = sorted(output_probs, reverse=True)
            if len(sorted_probs) < 2:
                return 0.0
            return 1.0 - (sorted_probs[0] - sorted_probs[1])  # Lower margin = higher uncertainty
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")

    def select_high_uncertainty_samples(self, samples: List[Dict[str, Any]], k: int = 10, method: str = "entropy") -> List[Dict[str, Any]]:
        """
        Given a list of samples with output probabilities, select top-k most uncertain.
        Each sample should be a dict with a 'probs' key (list of probabilities).
        """
        scored = [
            (self.uncertainty_score(s["probs"], method=method), s)
            for s in samples if "probs" in s
        ]
        scored.sort(reverse=True, key=lambda x: x[0])
        return [s for _, s in scored[:k]]

