from phi.logging import get_logger


def test_get_logger_emits_messages(caplog):
    logger = get_logger("phi.tests.logging")

    with caplog.at_level("INFO", logger=logger.name):
        logger.info("logging smoke test", extra={"scope": "unit"})

    assert "logging smoke test" in caplog.text
