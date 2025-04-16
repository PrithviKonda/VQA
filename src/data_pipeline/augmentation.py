"""
Image augmentation pipeline for VQA data pipeline.

- Uses albumentations for pixel-level and spatial-level augmentations.
- Augmentations referenced from architecture doc.

Author: VQA System Architect
"""

from typing import Optional, Callable
import albumentations as A
import numpy as np

def get_default_augmentation_pipeline(seed: Optional[int] = None) -> Callable:
    """
    Create an albumentations augmentation pipeline for VQA images.

    Pixel-level: brightness/contrast, noise, blur.
    Spatial-level: flips, rotations, crops.

    Args:
        seed: Optional random seed.

    Returns:
        Albumentations Compose object.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.MotionBlur(p=0.1),
        A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), p=0.5),
    ], p=1.0)


def apply_augmentation(image: np.ndarray, pipeline: Callable) -> np.ndarray:
    """
    Apply augmentation pipeline to image.

    Args:
        image: Input image as NumPy array.
        pipeline: Albumentations Compose object.

    Returns:
        Augmented image as NumPy array.
    """
    augmented = pipeline(image=image)
    return augmented["image"]


__all__ = [
    "get_default_augmentation_pipeline",
    "apply_augmentation"
]