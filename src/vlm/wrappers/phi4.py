
# src/vlm/wrappers/phi4.py

from src.vlm.inference import VLMBase
from src.data_pipeline.preprocessing import preprocess_image_phi4, preprocess_text_phi4
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from typing import Any

class Phi4VLM(VLMBase):
    """
    Wrapper for microsoft/Phi-4-multimodal-instruct using Hugging Face transformers.
    """
    def __init__(self, model_id: str, device: str = "cuda"):
        try:
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
            self.device = device
            self.model.to(self.device)
        except Exception as e:
            print(f"Error loading Phi-4 model: {e}")
            raise

    def predict(self, image: Any, question: str) -> str:
        try:
            image_inputs = preprocess_image_phi4(image)
            text_inputs = preprocess_text_phi4(question)
            inputs = self.processor(text=text_inputs, images=image_inputs, return_tensors="pt").to(self.device)
            output = self.model.generate(**inputs, max_new_tokens=100)
            answer = self.processor.decode(output[0], skip_special_tokens=True)
            return answer
        except Exception as e:
            print(f"Error during Phi-4 inference: {e}")
            return "Error: Unable to generate answer."