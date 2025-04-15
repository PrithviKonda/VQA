# src/api/models.py
from pydantic import BaseModel
from typing import Optional

class VQARequest(BaseModel):
    question: str
    # Image will be handled as file upload in FastAPI endpoint

class VQAResponse(BaseModel):
    question: str
    filename: str
    answer: str
    model: Optional[str] = None
