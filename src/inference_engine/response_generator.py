# src/inference_engine/response_generator.py
"""
Manages the multi-stage response process.
"""
from src.continuous_learning.active_learner import ActiveLearner

def generate_response(question: str, image, vlm, processor, retrieved_context: str = None, output_probs=None, *args, **kwargs):
    """
    Generate a response using the VLM, optionally augmented with retrieved context.
    Args:
        question: str
        image: PIL.Image
        vlm: VLM model instance
        processor: HuggingFace processor
        retrieved_context: Optional RAG context string
        output_probs: Optional list of VLM output probabilities
    Returns:
        dict: { 'answer': ..., 'uncertainty_score': ... }
    """
    from src.data_pipeline.preprocessing import preprocess_inputs
    prompt = retrieved_context if retrieved_context is not None else question
    inputs = preprocess_inputs(image, prompt, processor)
    # Call the VLM (forward pass)
    answer = vlm.generate_answer(inputs) if hasattr(vlm, 'generate_answer') else "[VLM output placeholder]"
    uncertainty_score = None
    if output_probs is not None:
        active_learner = ActiveLearner()
        uncertainty_score = active_learner.uncertainty_score(output_probs)
    return {
        "answer": answer,
        "uncertainty_score": uncertainty_score
    }
