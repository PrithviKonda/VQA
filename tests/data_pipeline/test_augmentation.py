"""
Unit tests for augmentation pipeline.
"""

import numpy as np
from src.data_pipeline.augmentation import get_default_augmentation_pipeline, apply_augmentation

def test_augmentation_pipeline_runs():
    img = np.ones((224, 224, 3), dtype=np.uint8) * 127
    aug = get_default_augmentation_pipeline(seed=42)
    out = apply_augmentation(img, aug)
    assert out.shape == (224, 224, 3)