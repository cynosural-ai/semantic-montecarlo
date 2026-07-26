"""
Log bootstrap for the pipeline.

Configures a single colored console handler on the root logger and silences the
noisy third-party libraries in this stack (``langchain`` logs every chain step
at INFO, ``httpx``/``httpcore`` log every request, ``openai`` logs retries).
Without suppressing these, every pipeline run drowns in transport logs.

Call :func:`setup_logging` explicitly from an entry point (CLI, script). It is
deliberately **not** invoked on package import — library code must not mutate
the host application's root logger as a side effect.
"""

import logging
import sys
from pathlib import Path

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[35m",  # Magenta
}
_RESET = "\033[0m"

# Third-party loggers that are noisy at INFO in this stack; quieted to WARNING
# so pipeline logs stay readable.
_NOISY_LIBRARIES = (
    "httpx",
    "httpcore",
    "openai",
)


class _ColorFormatter(logging.Formatter):
    """Colored formatter for console logging."""

    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, _RESET)
        record.levelname = f"{color}{record.levelname}{_RESET}"
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_file: Path | None = None,
) -> None:
    """
    Configure logging on the root logger.

    Always installs a colored console handler on stdout. When ``log_file`` is
    given, also installs a plain (uncolored) file handler so the same records
    are captured to disk — color codes would corrupt a log file. Replaces any
    existing root handlers.

    Args:
        log_level: Log level for application loggers (DEBUG, INFO, WARNING,
            ERROR, CRITICAL). Third-party libraries are pinned to WARNING
            regardless.
        log_file: Optional path to also write logs to. Parent directories are
            created if missing.
    """
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(
        _ColorFormatter(fmt=DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
    )

    handlers: list[logging.Handler] = [console]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter(fmt=DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
        )
        handlers.append(file_handler)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers = handlers

    for name in _NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(f"Logging configured (level={log_level})")


def get_logger(name: str) -> logging.Logger:
    """Return a logger with ``name`` (thin ``logging.getLogger`` passthrough)."""
    return logging.getLogger(name)
