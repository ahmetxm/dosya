from sim.opportunities import Leg, Opportunity
from sim.paper import PaperBroker


def _opp(cost_each: float = 0.40) -> Opportunity:
    shares = 10
    fee = 0.02
    return Opportunity(
        kind="binary_buy_pair",
        edge_per_share=0.16,
        locked=True,
        reason="test",
        event_title="t",
        shares=shares,
        expected_pnl=1.6,
        legs=[
            Leg("yes", "Yes", "buy", shares, cost_each, fee, -(shares * cost_each + fee), "m", "q"),
            Leg("no", "No", "buy", shares, cost_each, fee, -(shares * cost_each + fee), "m", "q"),
        ],
    )


def test_execute_updates_cash_and_positions():
    broker = PaperBroker(1000)
    trade = broker.execute(_opp())
    assert trade is not None
    assert broker.cash < 1000
    assert set(broker.positions) == {"yes", "no"}
    assert broker.positions["yes"].shares == 10


def test_mark_to_market_equity():
    broker = PaperBroker(1000)
    broker.execute(_opp())
    broker.update_marks({"yes": 0.50, "no": 0.50})
    assert broker.equity() == broker.cash + 10


def test_rejects_when_broke():
    broker = PaperBroker(1)
    assert broker.execute(_opp()) is None
    assert broker.rejected
