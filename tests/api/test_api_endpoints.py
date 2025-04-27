import io
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "message" in resp.json()

def test_vqa_endpoint(example_image):
    with open(example_image, "rb") as img_file:
        response = client.post(
            "/vqa/",
            data={"question": "What is in the image?"},
            files={"image_file": ("test.jpg", img_file, "image/jpeg")}
        )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data

def test_feedback_endpoint():
    payload = {
        "question": "What is in the image?",
        "image_id": "test.jpg",
        "generated_answer": "A cat on a sofa.",
        "user_rating": 5,
        "user_comment": "Correct and clear answer."
    }
    response = client.post("/feedback", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
