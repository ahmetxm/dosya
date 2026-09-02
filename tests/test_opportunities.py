from datetime import datetime, timezone

from sim.books import Level
from sim.opportunities import (
    QuotedMarket,
    find_binary_completeness,
    find_spot_certainty,
    find_strike_monotonicity,
    parse_market_meta,
)


def _quoted(**kwargs):
    defaults = dict(
        event_id="e1",
        event_title="Bitcoin above ___ today?",
        market={"id": "m1", "endDate": "2026-09-02T23:00:00Z"},
        question="Will the price of Bitcoin be above $70,000 on September 2?",
        yes_id="yes-1",
        no_id="no-1",
        yes_bids=[Level(0.48, 50)],
        yes_asks=[Level(0.49, 50)],
        no_bids=[Level(0.48, 50)],
        no_asks=[Level(0.49, 50)],
        fee_rate=0.07,
        min_size=5,
        asset="BTC",
        kind="above",
        strike=70000,
        end_date="2026-09-02T23:00:00Z",
    )
    defaults.update(kwargs)
    return QuotedMarket(**defaults)


def test_parse_reach_and_dip():
    assert parse_market_meta("Will Bitcoin reach $100,000 in September?") == ("BTC", "above", 100000.0)
    assert parse_market_meta("Will Bitcoin dip to $60,000 by December 31, 2026?") == ("BTC", "dip", 60000.0)


def test_binary_buy_pair_when_asks_sum_below_one():
    quoted = _quoted(yes_asks=[Level(0.40, 20)], no_asks=[Level(0.40, 20)])
    opps = find_binary_completeness(quoted)
    kinds = {item.kind for item in opps}
    assert "binary_buy_pair" in kinds
    buy = next(item for item in opps if item.kind == "binary_buy_pair")
    assert buy.locked is True
    assert buy.expected_pnl > 0
    assert buy.shares >= 5


def test_binary_sell_pair_when_bids_sum_above_one():
    quoted = _quoted(yes_bids=[Level(0.56, 20)], no_bids=[Level(0.56, 20)])
    opps = find_binary_completeness(quoted)
    assert any(item.kind == "binary_sell_pair" and item.expected_pnl > 0 for item in opps)


def test_no_binary_arb_on_a_normal_spread():
    quoted = _quoted()
    assert find_binary_completeness(quoted) == []


def test_monotonicity_inverted_ladder():
    easy = _quoted(
        market={"id": "low"},
        question="Will Bitcoin reach $80,000 in September?",
        yes_id="yes-low",
        strike=80000,
        yes_asks=[Level(0.40, 30)],
        yes_bids=[Level(0.38, 30)],
    )
    hard = _quoted(
        market={"id": "high"},
        question="Will Bitcoin reach $100,000 in September?",
        yes_id="yes-high",
        strike=100000,
        yes_asks=[Level(0.55, 30)],
        yes_bids=[Level(0.52, 30)],
    )
    opps = find_strike_monotonicity([easy, hard])
    assert opps
    assert opps[0].kind == "strike_monotonicity"
    assert opps[0].locked is True


def test_spot_certainty_buys_cheap_yes_when_already_through_strike():
    quoted = _quoted(
        yes_asks=[Level(0.90, 25)],
        strike=70000,
        end_date="2026-09-02T23:00:00Z",
    )
    opps = find_spot_certainty([quoted], {"BTC": 77100}, now=datetime(2026, 9, 2, 20, 0, tzinfo=timezone.utc))
    assert any(item.kind == "spot_yes_certainty" for item in opps)
