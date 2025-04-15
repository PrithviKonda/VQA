"""
Redis cache utilities for VQA system.
Phase 0: Basic get/set.
"""

import redis
from typing import Optional

_redis_client = None

def get_redis_client(host: str, port: int, db: int) -> redis.Redis:
    """
    Returns a singleton Redis client.
    """
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
    return _redis_client

class RedisCache:
    """
    Wrapper for Redis cache.
    """
    def __init__(self, client: redis.Redis):
        self.client = client

    def set(self, key: str, value: str, ex: Optional[int] = None) -> None:
        self.client.set(key, value, ex=ex)

    def get(self, key: str) -> Optional[str]:
        return self.client.get(key)

def get_cache_client(host: str = "localhost", port: int = 6379, db: int = 0) -> RedisCache:
    """
    Returns a RedisCache wrapper instance.
    """
    client = get_redis_client(host, port, db)
    return RedisCache(client)