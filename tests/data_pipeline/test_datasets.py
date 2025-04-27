import os
import tempfile
import json
from PIL import Image
from transformers import AutoProcessor
from src.data_pipeline.datasets import VQADataset

def make_dummy_jsonl_and_image(tmpdir):
    img_path = os.path.join(tmpdir, "img1.png")
    img = Image.new("RGB", (224, 224), color="red")
    img.save(img_path)
    jsonl_path = os.path.join(tmpdir, "data.jsonl")
    with open(jsonl_path, "w") as f:
        f.write(json.dumps({"image_path": img_path, "question": "What color?", "answers": ["red"]}) + "\n")
    return jsonl_path

import pytest

def test_vqa_dataset_loading(tmp_path):
    try:
        processor = AutoProcessor.from_pretrained("microsoft/phi-4-vision-128k-instruct")
    except Exception as e:
        pytest.skip(f"Model not available: {e}")
    jsonl_path = make_dummy_jsonl_and_image(str(tmp_path))
    ds = VQADataset(jsonl_path, processor, is_train=False)
    assert len(ds) == 1
    sample = ds[0]
    assert "pixel_values" in sample
    assert "input_ids" in sample
    assert "answers" in sample
    assert sample["answers"] == ["red"]
