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
from src.knowledge.retriever import TextRetriever
from src.knowledge.mrag import MRAGHandler

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    load_vlm_model()
    yield

app = FastAPI(
    title="Advanced Multimodal VQA System API",
    description="API endpoint for the VQA system based on Phi-4 Multimodal.",
    version="0.1.0",
    lifespan=lifespan
)

feedback_store = FeedbackStore()

@app.get("/", summary="Root endpoint", description="Simple health check endpoint.")
async def read_root():
    return {"message": "VQA System API is running."}

from fastapi import Query

# Instantiate retriever and handler (dummy index for now)
retriever = TextRetriever()
retriever.build_index(["Sample context about biology.", "Another fact about medicine."])
mrag_handler = MRAGHandler(retriever)

@app.post("/vqa/", response_model=VQAResponse, summary="Perform Visual Question Answering")
async def run_vqa_endpoint(
    question: str = Form(..., description="The question to ask about the image."),
    image_file: UploadFile = File(..., description="The image file to analyze."),
    use_rag: bool = Query(False, description="Enable Retrieval-Augmented Generation")
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
        prompt = mrag_handler.augment_prompt(question) if use_rag else question
        answer = perform_vqa(image, prompt)
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
        feedback_store.save_feedback(feedback.model_dump())
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save feedback: {e}")
