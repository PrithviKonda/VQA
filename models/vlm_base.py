from abc import ABC, abstractmethod
from typing import Any

class VLMBase(ABC):
    """
    Abstract base class for Vision-Language Models (VLMs)
    """
    @abstractmethod
    def predict(self, image: Any, question: str) -> str:
        """
        Given an image and a question, return the model's answer as a string.
        """
        pass
