# src/knowledge/planner.py
"""
Implements a simple rule-based or state-machine planner for multi-hop reasoning.
"""
from typing import List, Dict

class Planner:
    """
    Plans reasoning steps for complex VQA queries.
    """
    def __init__(self):
        self.state = "init"

    def plan(self, question: str, context: Dict) -> List[str]:
        """
        Generate a sequence of reasoning steps for a given question.
        For demonstration, uses a simple rule-based approach.
        """
        steps = []
        if not question:
            return ["invalid_question"]
        if "and" in question or "then" in question:
            steps.append("decompose_question")
        steps.append("retrieve_relevant_facts")
        steps.append("generate_final_answer")
        return steps
