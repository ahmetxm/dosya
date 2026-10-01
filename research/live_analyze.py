"""Honest test on live 1Hz order-book snapshots.

For each completed window: estimate outcome from Coinbase spot as
mean(spot over last 60s) >= mean(spot over 60s before open) (proxy for Chainlink 60s TWAP).
Model: at second s of the window, P(up) from a Gaussian on the projected final TWAP using realized
per-second vol. Trade only if model edge over the *ask* (plus fee) exceeds a threshold.
Fit nothing on the evaluation data: vol is estimated from the preceding 10 minutes only.
"""
import json, math, os
import numpy as np
import pandas as pd
from load import fee

D = os.path.join(os.path.dirname(__file__), "data")
df = pd.DataFrame([json.loads(l) for l in open(os.path.join(D, "live.jsonl"))])
df["sec"] = df.t.astype(int)
df = df.drop_duplicates("sec").set_index("sec").sort_index()
spot = df.spot
mk = json.load(open(os.path.join(D, "markets.json"))) if os.path.exists(os.path.join(D, "markets.json")) else []


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def outcomes():
    """Actual resolutions from Gamma for windows in the live log."""
    import requests
    res = {}
    for ts in sorted(df.ts.unique()):
        ev = requests.get("https://gamma-api.polymarket.com/events",
                          params={"slug": f"btc-updown-5m-{ts}"}, timeout=10).json()
        if ev and ev[0]["markets"][0].get("closed"):
            pr = json.loads(ev[0]["markets"][0]["outcomePrices"])
            if pr[0] in ("0", "1"):
                res[ts] = float(pr[0])
    return res


res = outcomes()
trades = []
calib = []
for ts, up in res.items():
    pre = spot.loc[ts - 60: ts - 1]
    hist = spot.loc[ts - 600: ts]
    if len(pre) < 50 or len(hist) < 300:
        continue
    ref = pre.mean()
    sig = np.log(hist).diff().std()  # per-second log vol
    for s in range(30, 296, 5):
        now = ts + s
        if now not in df.index:
            continue
        row = df.loc[now]
        x = spot.loc[ts + 240: now] if s > 240 else None
        cur = math.log(row.spot / ref)
        if s <= 240:
            # final TWAP over [240,300]: mean of path; variance of mean of BM over 60s starting h=240-s ahead
            h = 240 - s
            var = sig ** 2 * (h + 60 / 3)
            mu = cur
        else:
            k = s - 240
            done = math.log(x.mean() / ref) * k / 60
            r = 60 - k
            mu = done + cur * r / 60
            var = sig ** 2 * (r ** 3 / 3) / 3600
        q = phi(mu / math.sqrt(var)) if var > 0 else float(mu >= 0)
        calib.append((s, q, up))
        for side, ask, win, qq in [("up", row.up_ask, up, q), ("dn", row.dn_ask, 1 - up, 1 - q)]:
            if not ask:
                continue
            a, sz = ask
            edge = qq - a - fee(a)
            trades.append(dict(ts=ts, s=s, side=side, ask=a, size=sz, q=qq, edge=edge,
                               roi=(win - a - fee(a)) / a))

T = pd.DataFrame(trades)
C = pd.DataFrame(calib, columns=["s", "q", "up"])
print("windows resolved:", len(res), " snapshots:", len(df))
print("model Brier by phase:")
for lo, hi in [(30, 120), (120, 240), (240, 296)]:
    c = C[(C.s >= lo) & (C.s < hi)]
    print(f"  s[{lo},{hi}) brier={np.mean((c.q - c.up) ** 2):.4f} n={len(c)}")
print("\nOne trade per window (first signal), by edge threshold & phase:")
for thr in [0.02, 0.05, 0.1]:
    for lo, hi in [(30, 240), (240, 296)]:
        t = T[(T.edge > thr) & (T.s >= lo) & (T.s < hi)].groupby("ts").first()
        if len(t):
            print(f"  thr={thr:.2f} s[{lo},{hi}) n={len(t):3d} hit={np.mean(t.roi > 0):.3f} "
                  f"roi={t.roi.mean():+.4f} t={t.roi.mean() / (t.roi.std() + 1e-9) * np.sqrt(len(t)):+.2f}"
                  f" avg_ask={t.ask.mean():.3f} avg_size={t['size'].mean():.0f}")
