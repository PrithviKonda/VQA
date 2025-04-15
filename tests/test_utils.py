"""
Tests for utility functions.
"""

from src.utils.helpers import get_config

def test_get_config():
    config = get_config()
    assert "api" in config
    assert "vlm" in config
    assert "redis" in config