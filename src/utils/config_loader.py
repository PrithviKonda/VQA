import yaml
from pathlib import Path

def load_config(config_path='config.yaml'):
    """Load YAML config file from the project root."""
    config_path = Path(config_path)
    with config_path.open('r') as f:
        return yaml.safe_load(f)
