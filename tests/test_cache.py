import pytest
from src.cache.redis_cache import RedisCache
import os

@pytest.fixture(scope="module")
def redis_cache():
    config = {
        'redis_host': os.environ.get('REDIS_HOST', 'localhost'),
        'redis_port': int(os.environ.get('REDIS_PORT', 6379)),
        'redis_db': int(os.environ.get('REDIS_DB', 0)),
    }
    return RedisCache(config)

def test_redis_set_get(redis_cache):
    key = redis_cache.make_key('img', 'question')
    value = {"answer": "42"}
    redis_cache.set(key, value, expire=5)
    result = redis_cache.get(key)
    assert result == value
