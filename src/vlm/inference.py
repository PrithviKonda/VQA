"""
Inference logic for VLMs.
Phase 0: Basic Phi-4 inference.
"""

from typing import Any
from src.api.models import VQARequest
from src.vlm.loading import load_vlm

class Phi4VLMService:
    """
    Service for running inference with Phi-4 VLM.
    """
    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

    def answer_question(self, request: VQARequest) -> str:
        """
        Generates an answer for the given image and question.
        Phase 0: Text-only (image ignored).
        """
        prompt = f"Q: {request.question}\nA:"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=32)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract answer after "A:" for cleaner output
        if "A:" in answer:
            answer = answer.split("A:")[-1].strip()
        return answer

def infer_with_vlm(model_name: str, question: str, image):
    """
    Run inference using the specified VLM (e.g., LLaVA-NeXT).
    """
    vlm = load_vlm(model_name)
    if vlm is not None:
        return vlm.infer(question, image)
    return "VLM inference not available"