import requests

# Test /vqa/ endpoint (requires a running backend and a sample image)
def test_vqa():
    url = "http://localhost:8000/vqa/"
    files = {
        'question': (None, 'What is in the image?'),
        'image_file': ('sample.jpg', open('ar_frontend/public/sample.jpg', 'rb'), 'image/jpeg'),
    }
    response = requests.post(url, files=files)
    print("/vqa/ response:", response.status_code, response.text)

# Test /feedback endpoint
def test_feedback():
    url = "http://localhost:8000/feedback"
    payload = {
        "question": "What is in the image?",
        "image_id": "sample.jpg",
        "generated_answer": "A cat on a sofa.",
        "user_rating": 5,
        "user_comment": "Correct and clear answer."
    }
    response = requests.post(url, json=payload)
    print("/feedback response:", response.status_code, response.text)

if __name__ == "__main__":
    test_vqa()
    test_feedback()
