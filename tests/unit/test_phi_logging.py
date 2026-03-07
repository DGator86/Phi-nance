from pathlib import Path

from phi.logging import setup_logging


def test_setup_logging_writes_file(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "phi_test.log"
    logger = setup_logging(name="phi.test", log_level="INFO", log_file=log_file, console=False)

    logger.info("hello from test")

    assert log_file.exists()
    assert "hello from test" in log_file.read_text(encoding="utf-8")


def test_setup_logging_does_not_duplicate_handlers(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "dupe.log"
    logger = setup_logging(name="phi.dup", log_level="DEBUG", log_file=log_file, console=True)
    first = len(logger.handlers)

    logger = setup_logging(name="phi.dup", log_level="DEBUG", log_file=log_file, console=True)
    second = len(logger.handlers)

    assert second == first
