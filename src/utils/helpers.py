"""
Utility functions for VQA system.
Phase 0: Config loading.
"""

import yaml
from typing import Any, Dict
import threading

_config_cache: Dict[str, Any] = {}
_config_lock = threading.Lock()

def get_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Loads configuration from YAML file (singleton, thread-safe).
    """
    global _config_cache
    if not _config_cache:
        with _config_lock:
            if not _config_cache:
                with open(config_path, "r") as f:
                    _config_cache = yaml.safe_load(f)
    return _config_cache