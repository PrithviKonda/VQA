import pytest
from src.knowledge.retriever import TextRetriever

def test_text_retriever_build_and_retrieve(monkeypatch):
    # Mock SentenceTransformer.encode to return dummy embeddings
    class DummyModel:
        def get_sentence_embedding_dimension(self):
            return 2
        def encode(self, docs, convert_to_numpy=True, show_progress_bar=False):
            # Return unique vectors for each doc
            import numpy as np
            return np.array([[float(i), float(i+1)] for i in range(len(docs))], dtype=np.float32)
    monkeypatch.setattr("src.knowledge.retriever.SentenceTransformer", lambda name: DummyModel())
    retriever = TextRetriever()
    docs = ["foo", "bar", "baz"]
    retriever.build_index(docs)
    # Monkeypatch faiss.IndexFlatL2
    class DummyIndex:
        def add(self, arr): pass
        def search(self, arr, k): return None, [[0, 1, 2][:k]]
    monkeypatch.setattr("src.knowledge.retriever.faiss.IndexFlatL2", lambda dim: DummyIndex())
    retriever.index = DummyIndex()
    results = retriever.retrieve("test", top_k=2)
    assert isinstance(results, list)
    assert all(isinstance(r, str) for r in results)
