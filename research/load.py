import json, os
import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(__file__), "data")


def load():
    mk = json.load(open(os.path.join(D, "markets.json")))
    btc = pd.read_csv(os.path.join(D, "btc_1m.csv")).set_index("t")
    btc["mid"] = (btc.open + btc.high + btc.low + btc.close) / 4
    return mk, btc


def fee(p, rate=0.07):
    return rate * p * (1 - p)


def price_at(hist, t):
    """Last recorded Up price at or before absolute time t."""
    p = None
    for h in hist:
        if h["t"] <= t:
            p = h["p"]
        else:
            break
    return p
