import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
UI_STATE_PATH = Path(__file__).resolve().parent.parent / "config" / "ui_state.json"


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(override: Dict[str, Any] | None = None, config_path: Path | None = None) -> Dict[str, Any]:
    """
    Load configuration with overrides. If config_path is None, use default.
    """
    base_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    config = load_yaml(base_path)
    if override:
        config = deep_update(config, override)
    return config


def save_ui_state(state: Dict[str, Any]) -> None:
    UI_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(UI_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def load_ui_state() -> Dict[str, Any]:
    if not UI_STATE_PATH.exists():
        return {}
    return load_json(UI_STATE_PATH)


def deep_update(original: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update nested dictionaries.
    """
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(original.get(k), dict):
            original[k] = deep_update(original[k], v)
        else:
            original[k] = v
    return original


