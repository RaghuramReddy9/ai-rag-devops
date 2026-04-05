from pathlib import Path
from typing import Any

import yaml


def load_config(config_path:str = "configs/default.yaml") -> dict[str, Any]:
    """Load configuration from a YAML file."""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
