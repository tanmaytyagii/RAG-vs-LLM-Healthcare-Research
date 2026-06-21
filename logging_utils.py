"""
logging_utils.py
=================
Shared logging configuration used across all modules in the project.
Provides a single get_logger() factory so every module logs in a
consistent format to both stdout and a rotating log file.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

import config


def get_logger(name: str) -> logging.Logger:
    """Create (or retrieve) a configured logger.

    Args:
        name: Usually __name__ of the calling module.

    Returns:
        A logging.Logger instance with a stream handler and a rotating
        file handler attached exactly once (idempotent across repeated
        calls / module reloads).
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Already configured (e.g. Streamlit hot-reload); avoid duplicate handlers.
        return logger

    logger.setLevel(config.LOG_LEVEL)
    formatter = logging.Formatter(config.LOG_FORMAT)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    try:
        file_handler = RotatingFileHandler(
            config.LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # Filesystem may be read-only in some deployment environments;
        # stdout logging alone is acceptable in that case.
        logger.warning("Could not attach file handler; logging to stdout only.")

    logger.propagate = False
    return logger
