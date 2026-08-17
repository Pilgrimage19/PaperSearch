# ============================================================
# src/utils/logger.py - Logging setup
# ============================================================

import logging
import os
from pathlib import Path


def setup_logger(config) -> logging.Logger:
    """Setup logger from config, returns root logger."""
    log_config = config.get("logging", default={})
    level = getattr(logging, log_config.get("level", "INFO"), logging.INFO)
    log_file = log_config.get("file", "logs/pipeline.log")
    to_console = log_config.get("print_to_console", True)

    # Ensure log directory exists
    log_dir = Path(log_file).parent
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("paper_search")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    if to_console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
