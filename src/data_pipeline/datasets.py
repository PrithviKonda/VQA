# src/data_pipeline/datasets.py
"""
PyTorch Dataset classes and loader for VQA pipeline (placeholder for testing).
"""
import torch
from torch.utils.data import Dataset
from PIL import Image

def get_vqa_dataset(split="train", config=None):
    """
    Minimal placeholder VQA dataset for pipeline testing.
    Returns a torch.utils.data.Dataset instance with toy data.
    """
    class ToyVQADataset(Dataset):
        def __init__(self):
            # 2 fake samples
            self.samples = [
                {"image": Image.new("RGB", (224, 224), color="red"), "question": "What color is the image?", "answer": "red"},
                {"image": Image.new("RGB", (224, 224), color="blue"), "question": "What color is the image?", "answer": "blue"},
            ]
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx]
    return ToyVQADataset()
