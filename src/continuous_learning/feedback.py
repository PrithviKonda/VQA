# src/continuous_learning/feedback.py
"""
Handling and processing user feedback.
"""
import json
from typing import Dict, Any, Optional
from pathlib import Path

FEEDBACK_FILE = Path(__file__).parent / "user_feedback.jsonl"

class FeedbackStore:
    """
    Handles feedback storage in JSONL format.
    """
    """
    Simple file-based feedback storage (JSONL).
    """
    def __init__(self, file_path: Optional[Path] = None):
        self.file_path = file_path or FEEDBACK_FILE

    def save_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Append a feedback entry to the feedback file.
        """
        with open(self.file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(feedback) + '\n')
