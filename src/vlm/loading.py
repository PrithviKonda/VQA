"""
Model loading utilities for VLMs.
Phase 0: Load Phi-4 model.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Any

def load_phi4_model(model_name: str, device: str) -> Tuple[Any, Any]:
    """
    Loads the Phi-4 model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    return model, tokenizer