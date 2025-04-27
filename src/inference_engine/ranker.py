# src/inference_engine/ranker.py
"""
Ranks candidate answers using a judge model or scoring heuristic.
"""
from typing import List, Tuple

class Ranker:
    """
    Interface for ranking answers.
    """
    def __init__(self, judge_model=None):
        self.judge_model = judge_model

    def rank(self, candidates: List[str], context: dict) -> List[Tuple[str, float]]:
        """
        Rank candidates using the judge model or a heuristic.
        Returns a list of (answer, score), sorted by score descending.
        """
        if not candidates:
            return []
        if self.judge_model is not None:
            # Use model to score candidates
            scores = [self.judge_model.score(ans, context) for ans in candidates]
        else:
            # Simple heuristic: length of answer
            scores = [len(ans) for ans in candidates]
        return sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
