from sim.books import Level, best_ask, best_bid, walk_asks, walk_bids


def test_walk_asks_vwap():
    levels = [Level(0.40, 10), Level(0.41, 10)]
    fill = walk_asks(levels, 15)
    assert fill.shares == 15
    assert abs(fill.vwap - (10 * 0.40 + 5 * 0.41) / 15) < 1e-9
    assert fill.exhausted is False


def test_walk_bids_prefers_highest():
    levels = [Level(0.30, 8), Level(0.33, 4)]
    fill = walk_bids(levels, 6)
    assert fill.shares == 6
    assert abs(fill.vwap - (4 * 0.33 + 2 * 0.30) / 6) < 1e-9


def test_best_levels():
    levels = [Level(0.2, 1), Level(0.25, 1)]
    assert best_bid(levels) == 0.25
    assert best_ask(levels) == 0.2
