import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import requests
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT, exist_ok=True)
S = requests.Session()


def get(url, params=None, tries=5):
    for i in range(tries):
        try:
            r = S.get(url, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(2 ** i)
                continue
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(2 ** i)
    return None


def fetch_event(ts):
    ev = get("https://gamma-api.polymarket.com/events", {"slug": f"btc-updown-5m-{ts}"})
    if not ev:
        return None
    m = ev[0]["markets"][0]
    if not m.get("closed"):
        return None
    prices = json.loads(m["outcomePrices"])
    toks = json.loads(m["clobTokenIds"])
    hist = get("https://clob.polymarket.com/prices-history",
               {"market": toks[0], "startTs": ts - 600, "endTs": ts + 360, "fidelity": 1})
    return {
        "ts": ts,
        "up_won": float(prices[0]),
        "volume": float(m.get("volumeNum") or m.get("volume") or 0),
        "taker_fee": m.get("takerBaseFee"),
        "up_token": toks[0],
        "hist": (hist or {}).get("history", []),
    }


def collect_markets(days):
    now = int(time.time()) // 300 * 300 - 600
    tss = list(range(now - days * 86400, now, 300))
    with ThreadPoolExecutor(16) as ex:
        rows = [r for r in ex.map(fetch_event, tss) if r]
    with open(os.path.join(OUT, "markets.json"), "w") as f:
        json.dump(rows, f)
    print("markets", len(rows))
    return tss[0], now + 600


def collect_btc(start, end):
    out = []
    t = start - 3600
    while t < end:
        e = min(t + 300 * 60, end)
        c = get("https://api.exchange.coinbase.com/products/BTC-USD/candles",
                {"granularity": 60, "start": pd.Timestamp(t, unit="s").isoformat(),
                 "end": pd.Timestamp(e, unit="s").isoformat()})
        out += c or []
        t = e
        time.sleep(0.15)
    df = pd.DataFrame(out, columns=["t", "low", "high", "open", "close", "vol"])
    df = df.drop_duplicates("t").sort_values("t")
    df.to_csv(os.path.join(OUT, "btc_1m.csv"), index=False)
    print("btc rows", len(df))


if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    s, e = collect_markets(days)
    collect_btc(s, e)
