# src/api/main.py
"""
FastAPI app definition and endpoints. Refactored from original main.py.
"""
import io
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from src.api.models import VQARequest, VQAResponse, FeedbackRequest
from src.continuous_learning.feedback import FeedbackStore
from src.api.dependencies import get_config
from src.vlm.loading import load_vlm_model
from src.vlm.inference import perform_vqa

app = FastAPI(
    title="Advanced Multimodal VQA System API",
    description="API endpoint for the VQA system based on Phi-4 Multimodal.",
    version="0.1.0"
)

feedback_store = FeedbackStore()

@app.on_event("startup")
async def startup_event():
    load_vlm_model()

@app.get("/", summary="Root endpoint", description="Simple health check endpoint.")
async def read_root():
    return {"message": "VQA System API is running."}

@app.post("/vqa/", response_model=VQAResponse, summary="Perform Visual Question Answering")
async def run_vqa_endpoint(
    question: str = Form(..., description="The question to ask about the image."),
    image_file: UploadFile = File(..., description="The image file to analyze."),
):
    if not image_file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Invalid file type '{image_file.content_type}'. Please upload an image.")
    try:
        content = await image_file.read()
        image = Image.open(io.BytesIO(content))
        if image.mode != 'RGB':
            image = image.convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read or process image file: {e}")
    try:
        answer = perform_vqa(image, question)
        return VQAResponse(question=question, filename=image_file.filename, answer=answer, model="phi-4-multimodal")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

@app.post("/feedback", summary="Submit user feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Accepts user feedback for VQA answers and stores it.
    """
    try:
        feedback_store.save_feedback(feedback.dict())
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save feedback: {e}")
