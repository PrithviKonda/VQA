from src.inference_engine.response_generator import generate_response
from unittest.mock import Mock, patch
from PIL import Image

def test_generate_response_with_probs():
    output_probs = [0.7, 0.2, 0.1]
    question = "What is in the image?"
    image = Image.new("RGB", (224, 224))
    vlm = Mock()
    vlm.generate_answer.return_value = "cat"
    processor = Mock()
    with patch("src.data_pipeline.preprocessing.preprocess_inputs", return_value={"dummy": "data"}):
        result = generate_response(question, image, vlm, processor, output_probs=output_probs)
    assert "answer" in result
    assert "uncertainty_score" in result
    assert result["uncertainty_score"] is not None

def test_generate_response_no_probs():
    question = "What is in the image?"
    image = Image.new("RGB", (224, 224))
    vlm = Mock()
    vlm.generate_answer.return_value = "dog"
    processor = Mock()
    with patch("src.data_pipeline.preprocessing.preprocess_inputs", return_value={"dummy": "data"}):
        result = generate_response(question, image, vlm, processor)
    assert "answer" in result
    assert "uncertainty_score" in result
    assert result["uncertainty_score"] is None
