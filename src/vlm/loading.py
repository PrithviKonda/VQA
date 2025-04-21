# src/vlm/loading.py
"""
Handles loading of VLM models and processors.
"""
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from src.utils.config_loader import load_config

model = None
processor = None

def load_vlm_model():
    global model, processor
    config = load_config()
    model_id = config['vlm']['model_id']
    device = config['vlm']['device']
    torch_dtype = torch.float16 if device == 'cuda' else torch.float32
    print(f"Loading processor for {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print(f"Loading model for {model_id} on device {device}...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    ).to(device)
    model.eval()
    print(f"Model loaded: {model is not None}, Processor loaded: {processor is not None}")
    return model, processor
