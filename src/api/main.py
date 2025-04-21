"""
FastAPI main application for VQA system.
Phase 0: /health and /vqa endpoints.
"""

from fastapi import FastAPI, Depends, HTTPException
from src.api.models import VQARequest, VQAResponse
from src.api.dependencies import get_vlm_service, get_cache, get_response_generator
from src.utils.logging_config import setup_logging
from loguru import logger

setup_logging()

app = FastAPI(title="Advanced Multimodal VQA API", version="0.1.0")

@app.get("/health")
def health_check():
    """
    Health check endpoint.
    """
    logger.info("Health check requested")
    return {"status": "ok"}

from fastapi import Query
from src.knowledge.retriever import TextRetriever
from src.knowledge.mrag import MRAGHandler

@app.post("/vqa", response_model=VQAResponse)
def vqa_endpoint(
    request: VQARequest,
    response_generator: ResponseGenerator = Depends(get_response_generator)
):
    """
    VQA endpoint using the advanced hybrid ResponseGenerator.
    """
    result = response_generator.generate_response(request.question, request.image)
    return VQAResponse(**result)