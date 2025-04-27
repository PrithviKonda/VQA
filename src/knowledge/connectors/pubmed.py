# src/knowledge/connectors/pubmed.py
"""
Example PubMed connector placeholder.
"""
from .base_connector import BaseConnector
from typing import List
import os
import logging

try:
    from Bio import Entrez
except ImportError:
    Entrez = None

class PubMedConnector(BaseConnector):
    """
    Connector for querying PubMed using NCBI Entrez API.
    """

    def __init__(self, email: str = None, api_key: str = None):
        self.logger = logging.getLogger("PubMedConnector")
        self.email = email or os.getenv("NCBI_EMAIL")
        self.api_key = api_key or os.getenv("NCBI_API_KEY")
        if Entrez is not None:
            Entrez.email = self.email
            if self.api_key:
                Entrez.api_key = self.api_key

    def retrieve(self, query: str, max_results: int = 5) -> List[str]:
        if Entrez is None:
            raise ImportError("BioPython is required for PubMedConnector.")
        if not self.email:
            raise RuntimeError("NCBI_EMAIL must be set in environment or passed to PubMedConnector.")

        try:
            handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
            record = Entrez.read(handle)
            ids = record.get("IdList", [])
            snippets = []
            if ids:
                fetch_handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="text")
                abstracts = fetch_handle.read().split("\n\n")
                for abs_text in abstracts:
                    if abs_text.strip():
                        snippets.append(abs_text.strip())
            return snippets
        except Exception as e:
            self.logger.error(f"PubMed search failed: {e}")
            return []
