import pytest

@pytest.fixture(scope="session")
def example_image(tmp_path_factory):
    # Create a dummy image file for testing
    from PIL import Image
    img_path = tmp_path_factory.mktemp("data") / "test.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path)
    return img_path
