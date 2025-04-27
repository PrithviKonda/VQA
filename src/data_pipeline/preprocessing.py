# src/data_pipeline/preprocessing.py
"""
Image/text preprocessing for VQA using Hugging Face AutoProcessor (Phi-4).
"""
from typing import Any, Dict
from PIL import Image

def preprocess_inputs(image: Image.Image, text: str, processor: Any) -> Dict[str, Any]:
    """
    Preprocess image and text for Phi-4 VLM using Hugging Face processor.
    Args:
        image: PIL.Image
        text: str
        processor: HuggingFace processor (e.g., AutoProcessor)
    Returns:
        Dict with keys: pixel_values, input_ids, attention_mask, etc. (as required by model)
    """
    # Processor handles resizing, normalization, and text tokenization
    inputs = processor(
        images=image,
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    # Remove batch dimension for inference convenience
    for k, v in inputs.items():
        if hasattr(v, 'squeeze'):
            inputs[k] = v.squeeze(0)
    return inputs
