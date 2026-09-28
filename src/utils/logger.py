"""Logging configuration. Provided, complete - not part of any activity."""
import logging
import sys


def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """Set up and return a configured stdout logger."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    return logger
