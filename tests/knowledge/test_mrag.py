import pytest
from src.knowledge.mrag import MRAGHandler

class DummyRetriever:
    def retrieve(self, query_text, top_k=3):
        return ["doc1", "doc2"]

def test_mrag_augment_prompt():
    handler = MRAGHandler(DummyRetriever())
    question = "What is the capital of France?"
    prompt = handler.augment_prompt(question)
    assert prompt.startswith("Context:")
    assert "doc1" in prompt and "doc2" in prompt
    assert "Question: What is the capital of France?" in prompt


def test_mrag_empty_retrieval():
    class EmptyRetriever:
        def retrieve(self, query_text, top_k=3):
            return []
    handler = MRAGHandler(EmptyRetriever())
    question = "What is the capital of France?"
    prompt = handler.augment_prompt(question)
    assert prompt == f"Question: {question}"
