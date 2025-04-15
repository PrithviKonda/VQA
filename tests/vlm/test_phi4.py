"""
Basic test for Phi-4 VLM loading.
"""

from src.vlm.loading import load_phi4_model

def test_load_phi4_model():
    model, tokenizer = load_phi4_model("microsoft/phi-4", "cpu")
    assert model is not None
    assert tokenizer is not None