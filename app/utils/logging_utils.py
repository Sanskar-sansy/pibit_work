"""
Logging utilities using Rich for beautiful console output
and structured file logging.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Shared console instance for the whole app
CONSOLE = Console(
    theme=Theme(
        {
            "info": "cyan",
            "warning": "yellow",
            "error": "bold red",
            "success": "bold green",
            "prompt": "bold magenta",
            "score": "bold blue",
        }
    )
)

_loggers: dict[str, logging.Logger] = {}


def get_logger(
    name: str,
    level: str = "INFO",
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Get or create a named logger with Rich console handler and optional file handler.

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        log_dir: Directory for log files.
        log_file: Log filename. Defaults to '{name}.log'.

    Returns:
        Configured logging.Logger instance.
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # Rich console handler
    rich_handler = RichHandler(
        console=CONSOLE,
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        markup=True,
    )
    rich_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.addHandler(rich_handler)

    # Optional file handler
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fname = log_file or f"{name.replace('.', '_')}.log"
        file_path = Path(log_dir) / fname
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # files always get full detail
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger


def setup_root_logging(level: str = "INFO", log_dir: Optional[str] = None) -> None:
    """Configure root logger for the application."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root.handlers.clear()

    rich_handler = RichHandler(
        console=CONSOLE,
        show_time=True,
        show_path=True,
        rich_tracebacks=True,
    )
    root.addHandler(rich_handler)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            Path(log_dir) / "app.log", encoding="utf-8"
        )
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            )
        )
        root.addHandler(file_handler)
