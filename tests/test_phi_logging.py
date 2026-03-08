from __future__ import annotations

from phi import logging as phi_logging


def test_setup_logging_creates_file_and_reuses_handlers(tmp_path):
    log_file = tmp_path / "phi.log"

    logger = phi_logging.setup_logging("phi.tests.logger", log_file=log_file, console=False)
    logger.info("first")

    # Calling setup twice on same logger should not duplicate file handlers.
    phi_logging.setup_logging("phi.tests.logger", log_file=log_file, console=False)
    logger.info("second")

    text = log_file.read_text(encoding="utf-8")
    assert "first" in text
    assert "second" in text
    assert len(logger.handlers) == 1


def test_get_logger_prefixes_non_phi_names():
    logger = phi_logging.get_logger("tests.logging")
    assert logger.name == "phi.tests.logging"
