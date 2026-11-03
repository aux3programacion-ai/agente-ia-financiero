"""settings.py - Lee config/settings.yaml y expone get_setting().

Admite tres convenciones, todas equivalentes:

    get_setting('ml', 'xgboost')              # claves anidadas
    get_setting('ml.xgboost')                 # ruta con puntos
    get_setting('ml.xgboost', {})             # ruta + default posicional
    get_setting('risk_management', default={})  # ruta + default por keyword

Antes esta función trataba cualquier argumento posicional como clave, por lo que
`get_setting('ml.calibracion', {})` devolvia None y silenciosamente descartaba la
configuracion en todo el repo.
"""
from pathlib import Path

import yaml

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
    """
    Resuelve una clave de settings.yaml.

    El ultimo argumento posicional se trata como default si no es un string
    (compatibilidad con la llamada historica get_setting('ruta', {})).
    """
    args = list(keys)

    if args and not isinstance(args[-1], str):
        default = args.pop()

    cfg = _load()
    for k in args:
        for parte in str(k).split('.'):
            if not isinstance(cfg, dict):
                return default
            cfg = cfg.get(parte)
            if cfg is None:
                return default

    return cfg if cfg is not None else default