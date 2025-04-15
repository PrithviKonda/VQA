"""
FastAPI main application for VQA system.
Phase 0: /health and /vqa endpoints.
"""

from fastapi import FastAPI, Depends, HTTPException
from src.api.models import VQARequest, VQAResponse
from src.api.dependencies import get_vlm_service, get_cache
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

@app.post("/vqa", response_model=VQAResponse)
def vqa_endpoint(
    request: VQARequest,
    vlm_service=Depends(get_vlm_service),
    cache=Depends(get_cache)
):
    """
    Visual Question Answering endpoint.
    Text-only for Phase 0.
    """
    cache_key = f"vqa:{request.image_url}:{request.question}"
    cached = cache.get(cache_key)
    if cached:
        logger.info("Cache hit for key: {}", cache_key)
        return VQAResponse(answer=cached)
    try:
        logger.info("Cache miss for key: {}. Running inference.", cache_key)
        answer = vlm_service.answer_question(request)
        cache.set(cache_key, answer, ex=3600)
        return VQAResponse(answer=answer)
    except Exception as e:
        logger.error("VQA inference failed: {}", str(e))
        raise HTTPException(status_code=500, detail="VQA inference failed.")