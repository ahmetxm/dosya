"""Record ~1Hz snapshots of the live BTC 5m market order book and BTC spot for honest backtests."""
import json, os, time
import requests

OUT = os.path.join(os.path.dirname(__file__), "data", "live.jsonl")
S = requests.Session()
cache = {}


def tokens(ts):
    if ts not in cache:
        ev = S.get("https://gamma-api.polymarket.com/events",
                   params={"slug": f"btc-updown-5m-{ts}"}, timeout=5).json()
        cache[ts] = json.loads(ev[0]["markets"][0]["clobTokenIds"]) if ev else None
    return cache[ts]


def top(book):
    bids = sorted((float(b["price"]), float(b["size"])) for b in book.get("bids", []))
    asks = sorted((float(a["price"]), float(a["size"])) for a in book.get("asks", []))
    return (bids[-1] if bids else None), (asks[0] if asks else None)


with open(OUT, "a") as f:
    while True:
        t0 = time.time()
        ts = int(t0) // 300 * 300
        try:
            toks = tokens(ts)
            spot = float(S.get("https://api.exchange.coinbase.com/products/BTC-USD/ticker",
                               timeout=3).json()["price"])
            books = S.post("https://clob.polymarket.com/books",
                           json=[{"token_id": t} for t in toks], timeout=3).json()
            by = {b["asset_id"]: top(b) for b in books}
            ub, ua = by[toks[0]]
            db, da = by[toks[1]]
            f.write(json.dumps(dict(t=t0, ts=ts, spot=spot, up_bid=ub, up_ask=ua,
                                    dn_bid=db, dn_ask=da)) + "\n")
            f.flush()
        except Exception as e:
            print("err", e, flush=True)
        time.sleep(max(0, 1 - (time.time() - t0)))
