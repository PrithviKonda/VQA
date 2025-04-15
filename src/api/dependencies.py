"""
Dependency injection for API.
Phase 0: VLM service and Redis client.
"""

from functools import lru_cache
from src.vlm.loading import load_phi4_model
from src.vlm.inference import Phi4VLMService
from src.cache.redis_cache import get_redis_client
from src.utils.helpers import get_config

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