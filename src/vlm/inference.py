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
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="VLM Model not available.")
    config = load_config()
    prompt = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>"
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)
    generation_args = {
        "max_new_tokens": config['vlm'].get('max_new_tokens', 100),
        "temperature": config['vlm'].get('temperature', 0.7),
        "do_sample": True,
        "eos_token_id": processor.tokenizer.eos_token_id
    }
    with torch.no_grad():
        generate_ids = model.generate(**inputs, **generation_args)
    input_token_len = inputs["input_ids"].shape[1]
    response_ids = generate_ids[:, input_token_len:]
    response = processor.batch_decode(response_ids, skip_special_tokens=True)[0].strip()
    return response
