# src/knowledge/mrag.py
"""
MRAGHandler for Retrieval-Augmented Generation (RAG) in VQA.
"""
from typing import Any
from src.knowledge.retriever import TextRetriever

class MRAGHandler:
    def __init__(self, retriever: TextRetriever):
        self.retriever = retriever

    def augment_prompt(self, question: str, image: Any = None, top_k: int = 3) -> str:
        """
        Retrieve relevant context and prepend to prompt.
        """
        context_docs = self.retriever.retrieve(question, top_k=top_k)
        if context_docs:
            context_str = "Context:\n" + "\n- ".join([""] + context_docs) + f"\n\nQuestion: {question}"
        else:
            context_str = f"Question: {question}"
        return context_str
