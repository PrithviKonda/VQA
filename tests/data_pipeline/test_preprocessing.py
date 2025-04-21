"""
Unit tests for preprocessing functions.
"""

import numpy as np
from src.data_pipeline import preprocessing

def test_resize_and_normalize_image():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 127
    out = preprocessing.resize_and_normalize_image(img)
    assert out.shape == (224, 224, 3)
    assert np.isfinite(out).all()

def test_basic_text_preprocess():
    text = "  Hello World!  "
    out = preprocessing.basic_text_preprocess(text)
    assert out == "hello world!"

def test_tokenize_text_stub():
    text = "a b c"
    tokens = preprocessing.tokenize_text_stub(text)
    assert tokens == ["a", "b", "c"]