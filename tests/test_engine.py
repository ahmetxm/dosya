from sim.engine import EngineConfig, LiveEngine


class FakeClient:
    def fetch_spot(self):
        return {"BTC": 77100.0, "ETH": 2400.0}

    def list_crypto_events(self, limit=10):
        return [
            {
                "id": "1",
                "title": "Bitcoin above ___ on September 2?",
                "endDate": "2026-09-02T23:59:00Z",
                "markets": [
                    {
                        "id": "m1",
                        "question": "Will the price of Bitcoin be above $70,000 on September 2?",
                        "outcomes": '["Yes", "No"]',
                        "clobTokenIds": '["yes-token", "no-token"]',
                        "active": True,
                        "closed": False,
                        "acceptingOrders": True,
                        "enableOrderBook": True,
                        "volume24hr": 50000,
                        "orderMinSize": 5,
                        "feeSchedule": {"rate": 0.07},
                        "endDate": "2026-09-02T23:59:00Z",
                    }
                ],
            }
        ]

    def fetch_books(self, token_ids):
        return {
            "yes-token": {
                "bids": [{"price": "0.40", "size": "40"}],
                "asks": [{"price": "0.41", "size": "40"}],
            },
            "no-token": {
                "bids": [{"price": "0.40", "size": "40"}],
                "asks": [{"price": "0.41", "size": "40"}],
            },
        }


def test_cycle_executes_locked_binary_arb_on_fixture_books():
    engine = LiveEngine(client=FakeClient(), config=EngineConfig(starting_cash=10_000, allow_unlocked=False))
    result = engine.run_cycle()
    assert result["scanned"] == 1
    assert result["opportunities"] >= 1
    assert result["executed"] >= 1
    assert engine.broker.trades
    assert engine.broker.equity() != 10_000 or engine.broker.cash != 10_000


def test_snapshot_is_paper_only():
    engine = LiveEngine(client=FakeClient())
    snap = engine.snapshot()
    assert snap["mode"] == "paper"
    assert "No Polymarket orders" in snap["note"]
