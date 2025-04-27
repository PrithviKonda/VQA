from src.inference_engine.response_generator import generate_response

def test_generate_response_with_probs():
    output_probs = [0.7, 0.2, 0.1]
    result = generate_response(output_probs=output_probs)
    assert "answer" in result
    assert "uncertainty_score" in result
    assert result["uncertainty_score"] is not None

def test_generate_response_no_probs():
    result = generate_response()
    assert "answer" in result
    assert "uncertainty_score" in result
    assert result["uncertainty_score"] is None
