"""
PyTorch Dataset for VQA image-question-answer triplets.

- Loads data from JSON lines (each line: {image_path, question, answer}).
- Integrates preprocessing and augmentation.

Author: VQA System Architect
"""

import json
from typing import Callable, Optional, Dict, Any
from torch.utils.data import Dataset
import numpy as np
import cv2

from .preprocessing import resize_and_normalize_image, basic_text_preprocess

class VQADataset(Dataset):
    """
    Dataset for VQA tasks.

    Args:
        jsonl_path: Path to JSON lines file.
        image_root: Root directory for images.
        image_preprocess: Function to preprocess images.
        text_preprocess: Function to preprocess questions/answers.
        augmentation: Optional augmentation pipeline.
    """

    def __init__(
        self,
        jsonl_path: str,
        image_root: str,
        image_preprocess: Callable = resize_and_normalize_image,
        text_preprocess: Callable = basic_text_preprocess,
        augmentation: Optional[Callable] = None
    ):
        self.samples = []
        self.image_root = image_root
        self.image_preprocess = image_preprocess
        self.text_preprocess = text_preprocess
        self.augmentation = augmentation

        with open(jsonl_path, "r") as f:
            for line in f:
                item = json.loads(line)
                self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        img_path = f"{self.image_root}/{item['image_path']}"
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.augmentation:
            image = self.augmentation(image=image)["image"]
        image = self.image_preprocess(image)
        question = self.text_preprocess(item["question"])
        answer = self.text_preprocess(item["answer"])
        return {
            "image": image,
            "question": question,
            "answer": answer
        }


__all__ = ["VQADataset"]