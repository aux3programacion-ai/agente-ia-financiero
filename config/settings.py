"""settings.py - Lee config/settings.yaml y expone get_setting()."""
import os, yaml
from pathlib import Path

_CONFIG = None
_CONFIG_PATH = Path(__file__).parent / "settings.yaml"

def _load():
    global _CONFIG
    if _CONFIG is None:
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, encoding="utf-8") as f:
                _CONFIG = yaml.safe_load(f) or {}
        else:
            _CONFIG = {}
    return _CONFIG

def get_setting(*keys, default=None):
    cfg = _load()
    for k in keys:
        if isinstance(cfg, dict):
            cfg = cfg.get(k)
        else:
            return default
    return cfg if cfg is not None else default
