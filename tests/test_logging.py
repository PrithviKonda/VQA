import pytest
from src.utils.logging_config import setup_logging
import logging

def test_setup_logging(tmp_path, monkeypatch):
    # Create a temporary config.yaml
    config_content = '''
logging:
  level: DEBUG
  log_file: "test_vqa.log"
'''
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)
    monkeypatch.chdir(tmp_path)
    setup_logging(str(config_file))
    logger = logging.getLogger("test_logger")
    logger.debug("debug message")
    assert logging.getLogger().level == logging.DEBUG
