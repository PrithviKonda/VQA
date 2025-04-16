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

from fastapi import Query
from src.knowledge.retriever import TextRetriever
from src.knowledge.mrag import MRAGHandler
from src.inference_engine.response_generator import ResponseGenerator

@app.post("/vqa", response_model=VQAResponse)
def vqa_endpoint(
    request: VQARequest,
    vlm_service=Depends(get_vlm_service),
    cache=Depends(get_cache),
    use_rag: bool = Query(False, description="Enable retrieval-augmented generation (RAG)")
):
    """
    Visual Question Answering endpoint.
    Supports optional retrieval-augmented generation (RAG).
    """
    cache_key = f"vqa:{request.image_url}:{request.question}:rag={use_rag}"
    cached = cache.get(cache_key)
    if cached:
        logger.info("Cache hit for key: {}", cache_key)
        return VQAResponse(answer=cached)
    try:
        logger.info("Cache miss for key: {}. Running inference.", cache_key)
        retrieved_context = None
        if use_rag:
            retriever = TextRetriever()  # In real code, inject or reuse instance
            # retriever.load_vector_store('path/to/index')
            mrag = MRAGHandler(retriever)
            retrieved_context = mrag.augment_prompt(request.question)
        # Assume vlm_service is compatible with ResponseGenerator
        response_gen = ResponseGenerator(vlm=vlm_service)
        answer = response_gen.generate(
            question=request.question,
            image=request.image_url,  # Or pass image tensor/object as needed
            retrieved_context=retrieved_context
        )
        cache.set(cache_key, answer, ex=3600)
        return VQAResponse(answer=answer)
    except Exception as e:
        logger.error("VQA inference failed: {}", str(e))
        raise HTTPException(status_code=500, detail="VQA inference failed.")