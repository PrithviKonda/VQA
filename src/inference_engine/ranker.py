# src/inference_engine/ranker.py
"""
Ranks candidate answers using a judge model or scoring heuristic.
"""
from typing import List, Tuple, Dict, Optional
import logging
import os
import time

try:
    import openai
except ImportError:
    openai = None

class AnswerRanker:
    """
    Ranks candidate answers using heuristics or an external judge model (OpenAI GPT-3.5/4).
    """

    def __init__(self, use_external_judge: bool = False):
        self.use_external_judge = use_external_judge and openai is not None
        self.logger = logging.getLogger("AnswerRanker")
        self.max_retries = 3

    def rank_answers(
        self,
        candidates: List[str],
        context: Optional[Dict] = None,
        confidences: Optional[List[float]] = None
    ) -> List[Tuple[str, float]]:
        """
        Rank answers using either confidence scores, text heuristics, or external judge.
        """
        if not candidates:
            return []

        if self.use_external_judge:
            return self._rank_with_judge(candidates, context)
        elif confidences is not None and len(confidences) == len(candidates):
            # Use VLM confidence scores
            return sorted(zip(candidates, confidences), key=lambda x: x[1], reverse=True)
        else:
            # Simple heuristic: prefer longer, more detailed answers
            return sorted(
                ((ans, len(ans)) for ans in candidates),
                key=lambda x: x[1],
                reverse=True
            )

    def _rank_with_judge(self, candidates: List[str], context: Optional[Dict]) -> List[Tuple[str, float]]:
        """
        Use OpenAI GPT to score each answer.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment.")
        openai.api_key = api_key

        prompt = self._build_prompt(candidates, context)
        for attempt in range(self.max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=100
                )
                scores = self._parse_judge_response(response, candidates)
                return sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            except Exception as e:
                self.logger.warning(f"OpenAI judge call failed (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt)
        raise RuntimeError("Failed to get ranking from OpenAI judge after retries.")

    def _build_prompt(self, candidates: List[str], context: Optional[Dict]) -> str:
        q = context.get("question", "") if context else ""
        prompt = f"Given the question: \"{q}\", rank the following candidate answers from best to worst, giving a score from 1 (worst) to 10 (best) for each:\n"
        for idx, ans in enumerate(candidates):
            prompt += f"{idx+1}. {ans}\n"
        prompt += "Return a list of scores in order, one per line."
        return prompt

    def _parse_judge_response(self, response, candidates: List[str]) -> List[float]:
        text = response["choices"][0]["message"]["content"]
        scores = []
        for line in text.strip().splitlines():
            try:
                score = float(line.strip())
                scores.append(score)
            except Exception:
                continue
        # If parsing failed, fallback to equal scores
        if len(scores) != len(candidates):
            scores = [5.0] * len(candidates)
        return scores
