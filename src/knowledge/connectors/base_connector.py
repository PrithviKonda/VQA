"""
Base connector for external knowledge sources.
Phase 2: Placeholder.
"""

from abc import ABC, abstractmethod
from typing import List

class BaseKnowledgeConnector(ABC):
    """
    Abstract base class for external knowledge connectors.
    """
    @abstractmethod
    def search(self, query: str) -> List[str]:
        """
        Search the external knowledge source for relevant information.
        """
        pass