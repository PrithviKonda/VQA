"""
Image and text preprocessing utilities for VQA data pipeline.

- Image preprocessing (resize, normalize) tailored for Phi-4 Multimodal (compatible with Hugging Face processors).
- Basic text preprocessing/tokenization stubs.
- Ensures consistent preprocessing for training and inference.

Author: VQA System Architect
"""

from typing import Tuple, Optional, Callable
import cv2
import numpy as np

try:
    from transformers import AutoImageProcessor
except ImportError:
    AutoImageProcessor = None

# Phi-4 Multimodal image size (placeholder, adjust as needed)
PHI4_IMAGE_SIZE = (224, 224)
PHI4_MEAN = [0.5, 0.5, 0.5]
PHI4_STD = [0.5, 0.5, 0.5]


def resize_and_normalize_image(
    image: np.ndarray,
    size: Tuple[int, int] = PHI4_IMAGE_SIZE,
    mean: Optional[list] = None,
    std: Optional[list] = None
) -> np.ndarray:
    """
    Resize and normalize image for Phi-4 Multimodal.

    Args:
        image: Input image as a NumPy array (HWC, BGR or RGB).
        size: Target (width, height).
        mean: List of mean values for normalization.
        std: List of std values for normalization.

    Returns:
        Preprocessed image as float32 NumPy array.
    """
    if image is None:
        raise ValueError("Input image is None.")

    resized = cv2.resize(image, size)
    img = resized.astype(np.float32) / 255.0
    mean = mean or PHI4_MEAN
    std = std or PHI4_STD
    img = (img - mean) / std
    return img


def phi4_hf_preprocess(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image using Hugging Face's Phi-4 processor if available.

    Args:
        image: Input image as a NumPy array.

    Returns:
        Preprocessed image as NumPy array.
    """
    if AutoImageProcessor is None:
        raise ImportError("transformers not installed.")
    processor = AutoImageProcessor.from_pretrained("microsoft/phi-4-vision")
    return processor(image, return_tensors="np")["pixel_values"]


def basic_text_preprocess(text: str) -> str:
    """
    Basic text preprocessing for VQA (lowercase, strip).

    Args:
        text: Input text string.

    Returns:
        Preprocessed text string.
    """
    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")
    return text.lower().strip()


def tokenize_text_stub(text: str) -> list:
    """
    Stub for text tokenization (to be replaced with tokenizer integration).

    Args:
        text: Input text string.

    Returns:
        List of tokens (currently splits on whitespace).
    """
    return text.split()


# For tests: expose main functions
__all__ = [
    "resize_and_normalize_image",
    "phi4_hf_preprocess",
    "basic_text_preprocess",
    "tokenize_text_stub"
]