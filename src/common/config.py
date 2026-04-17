from pathlib import Path
from typing import Any

import yaml


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override values into a base config."""
    merged = dict(base)

    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value

    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: str = "configs/default.yaml") -> dict[str, Any]:
    """Load config, merging the base default config with an optional override file."""
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    default_path = Path("configs/default.yaml")
    if path == default_path:
        return _load_yaml(path)

    if not default_path.exists():
        raise FileNotFoundError(f"Default config file not found: {default_path}")

    base_config = _load_yaml(default_path)
    override_config = _load_yaml(path)
    return _merge_dicts(base_config, override_config)
