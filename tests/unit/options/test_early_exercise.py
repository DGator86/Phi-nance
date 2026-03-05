from phi.options.early_exercise import should_exercise_early


def test_call_no_dividend_not_early_exercise():
    assert should_exercise_early("call", 120, 100, 10 / 365, dividend_yield=0.0) is False


def test_put_itm_near_expiry_exercise():
    assert should_exercise_early("put", 90, 100, 10 / 365) is True


def test_put_otm_not_exercise():
    assert should_exercise_early("put", 110, 100, 10 / 365) is False
