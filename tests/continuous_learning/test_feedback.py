import tempfile
from src.continuous_learning.feedback import FeedbackStore

def test_save_feedback(tmp_path):
    store = FeedbackStore(file_path=tmp_path / "feedback.jsonl")
    feedback = {
        "question": "Q",
        "image_id": "id",
        "generated_answer": "A",
        "user_rating": 5,
        "user_comment": "Good"
    }
    store.save_feedback(feedback)
    with open(tmp_path / "feedback.jsonl") as f:
        lines = f.readlines()
    assert len(lines) == 1
    assert "Q" in lines[0]

def test_save_empty_feedback(tmp_path):
    store = FeedbackStore(file_path=tmp_path / "feedback_empty.jsonl")
    feedback = {}
    store.save_feedback(feedback)
    with open(tmp_path / "feedback_empty.jsonl") as f:
        lines = f.readlines()
    assert len(lines) == 1
    assert "{}" in lines[0]
