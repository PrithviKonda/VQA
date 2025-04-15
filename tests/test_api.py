"""
Basic tests for API endpoints.
"""

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_vqa_endpoint(monkeypatch):
    # Patch the VLM service to return a fixed answer for testing
    class DummyVLM:
        def answer_question(self, request):
            return "dummy answer"
    app.dependency_overrides = {}
    from src.api.dependencies import get_vlm_service
    app.dependency_overrides[get_vlm_service] = lambda: DummyVLM()
    payload = {
        "image_url": "[http://example.com/image.jpg",](http://example.com/image.jpg",)
        "question": "What is in the image?"
    }
    response = client.post("/vqa", json=payload)
    assert response.status_code == 200
    assert response.json()["answer"] == "dummy answer"
    app.dependency_overrides = {}