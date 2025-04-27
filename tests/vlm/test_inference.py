from src.vlm import inference

def test_perform_vqa_mock(example_image):
    # Should return mock answer if model is None
    from PIL import Image
    img = Image.open(example_image)
    answer = inference.perform_vqa(img, "What is in the image?")
    assert "mock" in answer
