# src/api/models.py
from pydantic import BaseModel
from typing import Optional

class FeedbackRequest(BaseModel):
    question: str
    image_id: Optional[str] = None
    generated_answer: str
    user_rating: int  # e.g., 1-5
    user_comment: Optional[str] = None

class VQARequest(BaseModel):
    question: str
    # Image will be handled as file upload in FastAPI endpoint

class VQAResponse(BaseModel):
    question: str
    filename: str
    answer: str
    model: Optional[str] = None
