from models.vlm_base import VLMBase
from typing import Any

# For demonstration, we use Hugging Face transformers and PIL
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch

class LlavaVLM(VLMBase):
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf", device: str = "cuda"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
        self.device = device
        self.model.to(self.device)

    def predict(self, image: Any, question: str) -> str:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        inputs = self.processor(text=question, images=image, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=128)
        answer = self.processor.decode(output[0], skip_special_tokens=True)
        return answer
