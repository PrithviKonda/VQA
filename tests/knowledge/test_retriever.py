"""
Unit tests for TextRetriever and MRAGHandler.
"""
import pytest
from src.knowledge.retriever import TextRetriever
from src.knowledge.mrag import MRAGHandler

class DummyVectorStore:
    def __init__(self):
        # 2 fake docs, 384-dim vectors
        self.embeddings = [
            [1.0] * 384,
            [0.0] * 384
        ]
        self.docs = ["Doc1 content.", "Doc2 content."]
    def search(self, query_vec, top_k):
        # Always return first k docs for test
        return [(self.docs[i], 0.9) for i in range(min(top_k, len(self.docs)))]

def test_text_retriever_embed():
    class DummyModel:
        def encode(self, x, **kwargs):
            return [[1.0] * 384]
    retriever = TextRetriever(model=DummyModel(), vector_store=DummyVectorStore())
    emb = retriever._embed("test")
    assert isinstance(emb, list)
    assert len(emb[0]) == 384

def test_text_retriever_retrieve():
    class DummyModel:
        def encode(self, x, **kwargs):
            return [[1.0] * 384]
    retriever = TextRetriever(model=DummyModel(), vector_store=DummyVectorStore())
    results = retriever.retrieve("test", top_k=2)
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)

def test_mraghandler_augment_prompt():
    class DummyRetriever:
        def retrieve(self, q, top_k):
            return ["Snippet1", "Snippet2"]
    mrag = MRAGHandler(retriever=DummyRetriever())
    prompt = mrag.augment_prompt("What is shown?")
    assert prompt.startswith("Context:")
    assert "Snippet1" in prompt and "Snippet2" in prompt
    assert "Question:" in prompt
