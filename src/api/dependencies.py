"""
Dependency injection for API.
Phase 0: VLM service and Redis client.
"""

from functools import lru_cache
from src.vlm.loading import load_phi4_model
from src.vlm.inference import Phi4VLMService
from src.cache.redis_cache import get_redis_client
from src.utils.helpers import get_config
from src.knowledge.planner import AdaptivePlanner
from src.inference_engine.response_generator import ResponseGenerator
from src.inference_engine.ranker import AnswerRanker

@lru_cache(maxsize=1)
def get_vlm_service():
    """
    Dependency for VLM service.
    Loads Phi-4 model for inference (singleton).
    """
    config = get_config()
    model, tokenizer = load_phi4_model(config["vlm"]["model_name"], config["vlm"]["device"])
    return Phi4VLMService(model, tokenizer)

@lru_cache(maxsize=1)
def get_cache():
    """
    Dependency for Redis cache client (singleton).
    """
    config = get_config()
    return get_redis_client(config["redis"]["host"], config["redis"]["port"], config["redis"]["db"])

@lru_cache(maxsize=1)
def get_planner():
    """
    Dependency for AdaptivePlanner (singleton).
    """
    return AdaptivePlanner()

@lru_cache(maxsize=1)
def get_ranker():
    """
    Dependency for AnswerRanker (singleton).
    """
    return AnswerRanker()

@lru_cache(maxsize=1)
def get_response_generator():
    """
    Dependency for ResponseGenerator (singleton).
    """
    return ResponseGenerator(planner=get_planner(), ranker=get_ranker())