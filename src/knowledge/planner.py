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

class AdaptivePlanner:
    """
    AdaptivePlanner for reasoning in VQA.
    Uses rule-based logic and basic state tracking to select tools and plan next steps.
    """

    def __init__(self):
        self.state: Dict[str, Any] = {}
        self.available_tools = ["vlm", "retriever", "pubmed_connector"]

    def analyze_question(self, question: str) -> Dict[str, Any]:
        """
        Analyze the question to extract key reasoning cues.
        """
        analysis = {
            "is_biomedical": any(kw in question.lower() for kw in ["disease", "patient", "treatment", "pubmed"]),
            "requires_external_knowledge": any(kw in question.lower() for kw in ["according to", "based on", "literature"]),
            "is_multistep": "and" in question.lower() or "then" in question.lower()
        }
        return analysis

    def plan_next_step(self, question: str, context: Dict[str, Any]) -> str:
        """
        Decide the next reasoning step/tool based on question analysis and current state.
        """
        analysis = self.analyze_question(question)
        self.state["last_analysis"] = analysis

        # Biomedical or literature-based question
        if analysis["is_biomedical"] or analysis["requires_external_knowledge"]:
            return "pubmed_connector"
        # Multistep question, try retriever
        if analysis["is_multistep"]:
            return "retriever"
        # Default: use VLM
        return "vlm"

    def select_tool(self, question: str, context: Dict[str, Any]) -> str:
        """
        Select the tool to use for the next step.
        """
        step = self.plan_next_step(question, context)
        if step in self.available_tools:
            return step
        return "vlm"  # Fallback

    def update_state(self, key: str, value: Any) -> None:
        self.state[key] = value

    def get_state(self) -> Dict[str, Any]:
        return self.state
