"""
Unit tests for VQADataset.
"""

import os
import json
import numpy as np
import cv2
import tempfile
from src.data_pipeline.datasets import VQADataset

def test_vqa_dataset_loads_sample():
    # Create dummy image
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "img.png")
        cv2.imwrite(img_path, img)
        # Create dummy JSONL
        jsonl_path = os.path.join(tmpdir, "data.jsonl")
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({
                "image_path": "img.png",
                "question": "What color?",
                "answer": "White"
            }) + "\n")
        # Test dataset
        ds = VQADataset(jsonl_path=jsonl_path, image_root=tmpdir)
        sample = ds[0]
        assert "image" in sample and "question" in sample and "answer" in sample
        assert sample["image"].shape == (224, 224, 3)