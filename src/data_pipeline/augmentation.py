# src/data_pipeline/augmentation.py
"""
Defines albumentations pipelines for VQA data augmentation.
"""
from typing import Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

def create_augmentation_pipeline(is_train: bool) -> Optional[A.Compose]:
    """
    Returns an albumentations.Compose pipeline for image augmentation.
    Args:
        is_train: If True, use strong augmentations. If False, minimal/no augmentation.
    Returns:
        Albumentations Compose pipeline or None.
    """
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.3),
            A.ToFloat(max_value=255.0),
        ])
    else:
        return None

def apply_augmentation(image: Image.Image, pipeline: Optional[A.Compose]) -> Image.Image:
    if pipeline is None:
        return image
    arr = np.array(image)
    augmented = pipeline(image=arr)
    image_aug = Image.fromarray((augmented["image"]).astype(np.uint8))
    return image_aug
