from sim.fees import buy_cost, fee_rate_from_market, sell_proceeds, taker_fee


def test_crypto_fee_matches_official_100_share_table():
    assert taker_fee(100, 0.50, 0.07) == 1.75
    assert taker_fee(100, 0.10, 0.07) == 0.63
    assert taker_fee(100, 0.90, 0.07) == 0.63


def test_extremes_and_invalid_are_zero():
    assert taker_fee(10, 0.0) == 0.0
    assert taker_fee(10, 1.0) == 0.0
    assert taker_fee(0, 0.4) == 0.0


def test_buy_and_sell_include_fee():
    assert buy_cost(10, 0.40, 0.07) > 4.0
    assert sell_proceeds(10, 0.40, 0.07) < 4.0


def test_fee_rate_from_market_schedule():
    assert fee_rate_from_market({"feeSchedule": {"rate": 0.07}}) == 0.07
    assert fee_rate_from_market({}) == 0.07
