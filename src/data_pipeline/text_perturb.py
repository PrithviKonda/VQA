# src/data_pipeline/text_perturb.py
"""
Functions for text perturbation/augmentation.
"""
import random

def synonym_replacement(text: str) -> str:
    """
    Replace a random word in the text with a synonym (mock implementation).
    For demonstration, just appends ' (augmented)' to the text.
    """
    # TODO: Replace with real synonym replacement logic using nltk/wordnet
    return text + " (augmented)"
