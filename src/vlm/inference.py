# src/vlm/inference.py
"""
Core VQA inference logic (calling models).
"""
from PIL import Image
from src.vlm.loading import model, processor
from src.utils.config_loader import load_config
import torch
from fastapi import HTTPException

def perform_vqa(image: Image.Image, question: str) -> str:
    """
    VQA inference for BLIP-2 (AutoModelForVision2Seq). Uses standard prompt and processor logic.
    """
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="VLM Model not available.")
    config = load_config()
    inputs = processor(images=image, text=question, return_tensors="pt").to(model.device)
    generation_args = {
        "max_new_tokens": config['vlm'].get('max_new_tokens', 100),
        "temperature": config['vlm'].get('temperature', 0.7),
        "do_sample": True,
    }
    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generation_args)
    response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return response
