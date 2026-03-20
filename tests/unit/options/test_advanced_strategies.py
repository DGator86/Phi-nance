from phi.options.strategies import ButterflySpread, CalendarSpread, Collar, CoveredCall, DiagonalSpread, ProtectivePut


def test_butterfly_has_three_legs_and_ratio():
    strat = ButterflySpread("call", 95, 100, 105, 0.2, quantity=1)
    legs = strat.legs()
    assert len(legs) == 3
    assert [l.quantity for l in legs] == [1, 2, 1]


def test_calendar_diagonal_structures():
    cal = CalendarSpread("call", 100, 0.1, 0.3, 2)
    dia = DiagonalSpread("put", 95, 90, 0.1, 0.3, 1)
    assert len(cal.legs()) == 2
    assert len(dia.legs()) == 2


def test_covered_protective_collar_structures():
    assert len(CoveredCall(105, 0.2, 1).legs()) == 1
    assert len(ProtectivePut(95, 0.2, 1).legs()) == 1
    assert len(Collar(95, 105, 0.2, 1).legs()) == 2
