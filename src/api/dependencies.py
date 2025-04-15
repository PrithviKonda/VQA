# src/api/dependencies.py
"""
FastAPI dependency injection functions (e.g., DB session, config, cache).
"""
from src.utils.config_loader import load_config

def get_config():
    return load_config()
