# src/data_pipeline/datasets.py
"""
PyTorch Dataset class for VQA pipeline. Loads JSONL, applies augmentation and preprocessing.
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from typing import Any, Dict, Optional
from src.data_pipeline.augmentation import create_augmentation_pipeline, apply_augmentation
from src.data_pipeline.preprocessing import preprocess_inputs

class VQADataset(Dataset):
    def __init__(self, jsonl_path: str, processor: Any, is_train: bool = True):
        self.jsonl_path = jsonl_path
        self.processor = processor
        self.is_train = is_train
        self.augmentation = create_augmentation_pipeline(is_train)
        # Preload line offsets for random access
        self.line_offsets = []
        with open(jsonl_path, 'r') as f:
            offset = 0
            for line in f:
                self.line_offsets.append(offset)
                offset += len(line.encode('utf-8'))
        self.num_samples = len(self.line_offsets)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        with open(self.jsonl_path, 'r') as f:
            f.seek(self.line_offsets[idx])
            line = f.readline()
            sample = json.loads(line)
        image_path = sample["image_path"]
        question = sample["question"]
        answers = sample.get("answers") or sample.get("answer")
        image = Image.open(image_path).convert('RGB')
        image = apply_augmentation(image, self.augmentation)
        model_inputs = preprocess_inputs(image, question, self.processor)
        model_inputs["answers"] = answers
        return model_inputs
