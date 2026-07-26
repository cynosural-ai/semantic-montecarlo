"""
Tests for the logging setup.

Pins the load-bearing behavior: handlers on the root logger, the requested app
level, the third-party libraries quieted to WARNING, and the optional file
handler capturing records to disk.
"""

from __future__ import annotations

import logging
from pathlib import Path

from semantic_montecarlo.observability import get_logger, setup_logging


def test_setup_console_only_when_no_file() -> None:
    setup_logging("INFO")
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], logging.StreamHandler)
    assert not isinstance(root.handlers[0], logging.FileHandler)


def test_setup_adds_file_handler_when_log_file_given(tmp_path: Path) -> None:
    setup_logging("INFO", log_file=tmp_path / "run.log")
    root = logging.getLogger()
    assert len(root.handlers) == 2
    file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1


def test_setup_creates_parent_dirs_for_log_file(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "run.log"
    setup_logging("INFO", log_file=nested)
    assert nested.parent.is_dir()


def test_file_handler_captures_records(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    setup_logging("INFO", log_file=log_path)
    get_logger("semantic_montecarlo.test").info("captured record")
    # FileHandler buffers; flush before reading.
    for h in logging.getLogger().handlers:
        h.flush()
    contents = log_path.read_text(encoding="utf-8")
    assert "captured record" in contents


def test_setup_sets_app_log_level() -> None:
    setup_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG

    setup_logging("WARNING")
    assert logging.getLogger().level == logging.WARNING


def test_noisy_libraries_quieted_to_warning() -> None:
    # Regardless of the app level, these must stay at WARNING so transport
    # noise doesn't drown out pipeline logs.
    setup_logging("DEBUG")
    for name in ("langchain", "httpx", "httpcore", "openai"):
        assert logging.getLogger(name).level == logging.WARNING


def test_get_logger_returns_named_logger() -> None:
    logger = get_logger("semantic_montecarlo.test")
    assert logger.name == "semantic_montecarlo.test"
    assert isinstance(logger, logging.Logger)
