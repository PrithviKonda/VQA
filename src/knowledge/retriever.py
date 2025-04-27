# src/knowledge/retriever.py
"""
TextRetriever for RAG: embeds docs, builds FAISS index, retrieves top-k relevant docs.
"""
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

class TextRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents: List[str] = []
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def build_index(self, docs: List[str]):
        self.documents = docs
        if not docs:
            self.index = None
            return
        embeddings = self.model.encode(docs, convert_to_numpy=True, show_progress_bar=False)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings.astype(np.float32))

    def retrieve(self, query_text: str, top_k: int = 3) -> List[str]:
        if self.index is None or not self.documents:
            return []
        query_emb = self.model.encode([query_text], convert_to_numpy=True)
        D, I = self.index.search(query_emb.astype(np.float32), min(top_k, len(self.documents)))
        return [self.documents[i] for i in I[0] if i < len(self.documents)]
