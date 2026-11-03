#!/usr/bin/env python3
"""
logger.py - Structured logging for the Agente Financiero.
Replaces all print() calls with structured, level-aware logging.
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False

DATA_DIR = Path(__file__).parent
LOG_DIR = DATA_DIR / 'Datos' / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _setup_structlog(level: str = 'INFO'):
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt='iso'),
            structlog.dev.ConsoleRenderer() if sys.stderr.isatty() else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger()


def _setup_stdlib(level: str = 'INFO'):
    logger = logging.getLogger('agente_financiero')
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    logger.addHandler(console)

    log_file = LOG_DIR / f'agente_{datetime.now().strftime("%Y%m%d")}.log'
    file_handler = logging.FileHandler(str(log_file), encoding='utf-8')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


_global_logger: object | None = None


def get_logger(module_name: str = 'agente_financiero'):
    global _global_logger
    if _global_logger is not None:
        if HAS_STRUCTLOG:
            return _global_logger.bind(module=module_name)
        child = _global_logger.getChild(module_name)
        return child

    level = 'INFO'
    if HAS_STRUCTLOG:
        _global_logger = _setup_structlog(level)
        return _global_logger.bind(module=module_name)
    else:
        _global_logger = _setup_stdlib(level)
        child = _global_logger.getChild(module_name)
        return child


class LogAdapter:
    def __init__(self, module_name: str):
        self._log = get_logger(module_name)

    def _msg(self, msg: str, **kwargs):
        if HAS_STRUCTLOG:
            return self._log, kwargs
        return self._log, kwargs

    def info(self, msg: str, **kwargs):
        if HAS_STRUCTLOG:
            self._log.info(msg, **kwargs)
        else:
            self._log.info(msg)

    def warning(self, msg: str, **kwargs):
        if HAS_STRUCTLOG:
            self._log.warning(msg, **kwargs)
        else:
            self._log.warning(msg)

    def error(self, msg: str, **kwargs):
        if HAS_STRUCTLOG:
            self._log.error(msg, **kwargs)
        else:
            self._log.error(msg)

    def debug(self, msg: str, **kwargs):
        if HAS_STRUCTLOG:
            self._log.debug(msg, **kwargs)
        else:
            self._log.debug(msg)

    def critical(self, msg: str, **kwargs):
        if HAS_STRUCTLOG:
            self._log.critical(msg, **kwargs)
        else:
            self._log.critical(msg)


def get_module_logger(name: str) -> LogAdapter:
    return LogAdapter(name)
