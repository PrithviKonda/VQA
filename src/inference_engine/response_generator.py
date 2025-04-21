"""
Response generator for VQA answers.
Phase 3: Placeholder.
"""

import logging
from typing import Any, List, Dict
from src.knowledge.planner import AdaptivePlanner, SubQuestion
from src.knowledge.sebe_vqa import get_aligned_retriever, reselect_knowledge
from src.inference_engine.ranker import AnswerRanker
#from src.vlm.wrappers.llava import LLaVAWrapper  # Uncomment if LLaVA is enabled

class ResponseGenerator:
    """
    Orchestrates the hybrid inference pipeline: planning, retrieval, VLM, connectors, SeBe-VQA, and ranking.
    """
    def __init__(self, planner: AdaptivePlanner = None, ranker: AnswerRanker = None):
        self.planner = planner or AdaptivePlanner()
        self.ranker = ranker or AnswerRanker()
        self.retriever = get_aligned_retriever()
        # self.llava = LLaVAWrapper()  # Uncomment if LLaVA is enabled

    def generate_response(self, question: str, image: Any) -> Dict[str, Any]:
        """
        Main entry point for generating an answer to a VQA query.
        """
        logging.info(f"Generating response for question: {question}")
        sub_questions = self.planner.decompose_question(question, image)
        knowledge_candidates = []
        for sq in sub_questions:
            tool = self.planner.select_tool(sq)
            if tool == "retriever":
                knowledge = self.retriever(sq.text, image)
            elif tool == "pubmed_connector":
                from src.knowledge.connectors.pubmed import PubMedConnector
                connector = PubMedConnector()
                knowledge = connector.search(sq.text)
            elif tool == "vlm":
                # Placeholder: Call VLM (e.g., Phi-4 or LLaVA)
                knowledge = ["VLM answer stub"]
            else:
                knowledge = ["Unknown tool"]
            knowledge_candidates.extend(knowledge)

        # SeBe-VQA: knowledge filtering/re-selection
        filtered_candidates = reselect_knowledge(knowledge_candidates, question)

        # Generate candidate answers (simulate multiple answers)
        candidate_answers = [f"Answer based on: {kc}" for kc in filtered_candidates]

        # Rank answers
        ranked = self.ranker.rank_answers(question, image, candidate_answers)

        return {
            "answers": ranked,
            "candidates": candidate_answers,
            "knowledge": filtered_candidates
        }