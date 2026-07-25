"""Tests for the logging setup.

Pins the load-bearing behavior: a single handler on the root logger, the
requested app level, and the third-party libraries quieted to WARNING.
"""

from __future__ import annotations

import logging

from semantic_montecarlo.observability import get_logger, setup_logging


def test_setup_configures_root_with_single_handler() -> None:
    setup_logging("INFO")
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], logging.StreamHandler)


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
