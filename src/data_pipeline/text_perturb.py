"""
Text perturbation utilities for VQA data pipeline.

- Character-level and word-level perturbations:
    - Synonym replacement (stub)
    - Random insertion/deletion

Author: VQA System Architect
"""

import random
from typing import List

def random_deletion(text: str, p: float = 0.1) -> str:
    """
    Randomly delete characters from text.

    Args:
        text: Input string.
        p: Probability of deleting each character.

    Returns:
        Perturbed string.
    """
    return ''.join([c for c in text if random.random() > p])


def random_insertion(text: str, chars: str = "abcdefghijklmnopqrstuvwxyz", n: int = 1) -> str:
    """
    Randomly insert characters into text.

    Args:
        text: Input string.
        chars: Characters to insert from.
        n: Number of insertions.

    Returns:
        Perturbed string.
    """
    text = list(text)
    for _ in range(n):
        idx = random.randint(0, len(text))
        char = random.choice(chars)
        text.insert(idx, char)
    return ''.join(text)


def synonym_replacement_stub(text: str) -> str:
    """
    Stub for synonym replacement (to be implemented with external resources).

    Args:
        text: Input string.

    Returns:
        String with "synonyms" replaced (currently unchanged).
    """
    # Placeholder: no actual synonym replacement
    return text


def random_word_shuffle(text: str) -> str:
    """
    Shuffle words in the text randomly.

    Args:
        text: Input string.

    Returns:
        String with words shuffled.
    """
    words = text.split()
    random.shuffle(words)
    return ' '.join(words)


__all__ = [
    "random_deletion",
    "random_insertion",
    "synonym_replacement_stub",
    "random_word_shuffle"
]