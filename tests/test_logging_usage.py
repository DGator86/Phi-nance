from phi.logging import get_logger, setup_logging


def test_get_logger_emits_messages(capsys):
    logger = setup_logging("phi.tests.logging", log_level="INFO", console=True, log_file=None)
    logger.info("logging smoke test")
    captured = capsys.readouterr()
    assert "logging smoke test" in captured.out
