"""Logging configuration for Frame library."""

import logging
import time
from contextlib import contextmanager
from typing import Generator

import structlog


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a structlog logger."""
    return structlog.get_logger(name)


def configure_logging(
    level: str = "WARNING",
    json_output: bool = False,
) -> None:
    """Configure Frame logging.

    Args:
        level: Log level ("DEBUG", "INFO", "WARNING", "ERROR")
        json_output: True for JSON output (production), False for console
    """
    logging.basicConfig(format="%(message)s", level=getattr(logging, level.upper()))

    shared_processors = [
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_output:
        processors = shared_processors + [structlog.processors.JSONRenderer()]
    else:
        processors = shared_processors + [structlog.dev.ConsoleRenderer(colors=True)]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


@contextmanager
def timed_block(
    logger: structlog.BoundLogger,
    event: str,
    level: str = "debug",
) -> Generator[None, None, None]:
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        getattr(logger, level)(event, elapsed_ms=round(elapsed_ms, 2))
