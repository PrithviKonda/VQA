"""
Synthetic data generation concepts for VQA.

- CoSyn: Code synthesis, rendering, QA generation (all stubs).
- BiomedCLIP+LLM: Medical VQA pair generation (all stubs).

Author: VQA System Architect
"""

from typing import Any, Dict, List

# --- CoSyn Concept ---

def generate_code_stub(prompt: str) -> str:
    """
    Stub for LLM-based code generation.

    Args:
        prompt: Prompt for code synthesis.

    Returns:
        Generated code (stub).
    """
    # Placeholder: returns a dummy code string
    return "# generated code"


def render_image_stub(code: str) -> Any:
    """
    Stub for rendering image from code.

    Args:
        code: Code to render.

    Returns:
        Rendered image (stub).
    """
    return None  # Placeholder for rendered image


def generate_qa_stub(image: Any) -> Dict[str, str]:
    """
    Stub for LLM-based question-answer generation for an image.

    Args:
        image: Image object.

    Returns:
        Dict with 'question' and 'answer' (stub).
    """
    return {"question": "What is shown?", "answer": "A placeholder object."}


def generate_cosyn_synthetic_sample(prompt: str) -> Dict[str, Any]:
    """
    Generate a synthetic VQA sample using CoSyn concept.

    Args:
        prompt: Prompt for code synthesis.

    Returns:
        Dict with image, question, answer.
    """
    code = generate_code_stub(prompt)
    image = render_image_stub(code)
    qa = generate_qa_stub(image)
    return {"image": image, "question": qa["question"], "answer": qa["answer"]}


# --- BiomedCLIP+LLM Concept ---

def generate_biomedclip_vqa_stub(seed_text: str) -> Dict[str, Any]:
    """
    Stub for generating medical VQA pair using BiomedCLIP and LLM.

    Args:
        seed_text: Seed text for VQA generation.

    Returns:
        Dict with image, question, answer.
    """
    # Placeholder: no actual model inference
    return {
        "image": None,
        "question": "What is the abnormality?",
        "answer": "No abnormality detected."
    }


__all__ = [
    "generate_cosyn_synthetic_sample",
    "generate_biomedclip_vqa_stub"
]