"""
Logging configuration for VQA system.
Phase 0: Sets up loguru logging.
"""

import logging
from loguru import logger
import sys
from src.utils.helpers import get_config
import os

def setup_logging():
    """
    Configures logging using loguru.
    """
    config = get_config()
    log_file = config["logging"].get("log_file", "logs/vqa.log")
    level = config["logging"].get("level", "INFO")
    logger.remove()
    logger.add(sys.stdout, level=level)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger.add(log_file, level=level, rotation="10 MB", retention="10 days")
    logging.basicConfig(level=level)