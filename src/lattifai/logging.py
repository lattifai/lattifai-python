"""Unified logging configuration for LattifAI."""

import logging
import sys
from typing import Optional

# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
SIMPLE_FORMAT = "%(levelname)s: %(message)s"


def setup_logger(
    name: str,
    level: Optional[int] = None,
    format_string: Optional[str] = None,
    handler: Optional[logging.Handler] = None,
) -> logging.Logger:
    """
    Setup logger with consistent formatting for LattifAI modules.

    Args:
        name: Logger name (will be prefixed with 'lattifai.')
        level: Logging level (defaults to INFO)
        format_string: Custom format string (defaults to SIMPLE_FORMAT)
        handler: Custom handler (defaults to StreamHandler)

    Returns:
        Configured logger instance

    Examples:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Processing started")

        >>> logger = setup_logger("alignment", level=logging.DEBUG)
        >>> logger.debug("Debug information")
    """
    # Ensure name is prefixed with 'lattifai.'
    if not name.startswith("lattifai."):
        logger_name = f"lattifai.{name}"
    else:
        logger_name = name

    logger = logging.getLogger(logger_name)

    # Set level
    if level is None:
        level = logging.INFO
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Setup handler
    if handler is None:
        handler = logging.StreamHandler(sys.stderr)

    # Setup formatter
    if format_string is None:
        format_string = SIMPLE_FORMAT
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get existing logger or create new one with default settings.

    Args:
        name: Logger name (will be prefixed with 'lattifai.')

    Returns:
        Logger instance
    """
    if not name.startswith("lattifai."):
        logger_name = f"lattifai.{name}"
    else:
        logger_name = name

    logger = logging.getLogger(logger_name)

    # If logger has no handlers, set it up with defaults
    if not logger.handlers:
        return setup_logger(name)

    return logger


def set_log_level(level: int) -> None:
    """
    Set log level for all LattifAI loggers.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)

    Examples:
        >>> from lattifai.logging import set_log_level
        >>> import logging
        >>> set_log_level(logging.DEBUG)
    """
    root_logger = logging.getLogger("lattifai")
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)


__all__ = [
    "setup_logger",
    "get_logger",
    "set_log_level",
    "DEFAULT_FORMAT",
    "SIMPLE_FORMAT",
]
