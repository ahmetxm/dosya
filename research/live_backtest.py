"""Paper-trade the TWAP model against recorded best asks. One entry per window (first signal) to avoid overlap."""
import json, os, sys
from math import erf, sqrt
import numpy as np
import pandas as pd
import requests
from load import fee

D = os.path.join(os.path.dirname(__file__), "data")
snaps = pd.DataFrame([json.loads(l) for l in open(os.path.join(D, "live.jsonl"))])
snaps["ts"] = snaps.ts.astype(int)
spot = snaps.set_index("t").spot
lr = np.log(spot).diff().dropna()

outcomes_path = os.path.join(D, "live_outcomes.json")
outcomes = json.load(open(outcomes_path)) if os.path.exists(outcomes_path) else {}
for ts in sorted(snaps.ts.unique()):
    if str(ts) in outcomes:
        continue
    ev = requests.get("https://gamma-api.polymarket.com/events",
                      params={"slug": f"btc-updown-5m-{ts}"}, timeout=10).json()
    m = ev[0]["markets"][0] if ev else None
    if m and m.get("closed") and m.get("umaResolutionStatus") == "resolved":
        outcomes[str(ts)] = float(json.loads(m["outcomePrices"])[0])
json.dump(outcomes, open(outcomes_path, "w"))


def ncdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def model_prob(t, ts, s_now, sig):
    ref_w = spot[(spot.index >= ts - 60) & (spot.index < ts)]
    if len(ref_w) < 50:
        return None
    ref = ref_w.mean()
    a, b = ts + 240, ts + 300
    if t < a:
        mean, var = s_now, sig ** 2 * ((a - t) + (b - a) / 3)
    else:
        obs = spot[(spot.index >= a) & (spot.index <= t)]
        w = (t - a) / 60
        mean = w * obs.mean() + (1 - w) * s_now
        var = (1 - w) ** 2 * sig ** 2 * (b - t) / 3
    sd = sqrt(max(var, 1e-12)) * s_now
    return ncdf((mean - ref) / sd)


THR = [float(x) for x in sys.argv[1:]] or [0.0, 0.03, 0.06, 0.1]
rows = []
for ts, g in snaps.groupby("ts"):
    if str(ts) not in outcomes:
        continue
    up = outcomes[str(ts)] > 0.5
    for _, r in g.iterrows():
        if r.t > ts + 295 or r.up_ask is None or r.dn_ask is None:
            continue
        hist = lr[(lr.index > r.t - 600) & (lr.index <= r.t)]
        if len(hist) < 300:
            continue
        sig = hist.std()
        q = model_prob(r.t, ts, r.spot, sig)
        if q is None:
            continue
        ua, da = r.up_ask[0], r.dn_ask[0]
        rows.append(dict(ts=ts, el=r.t - ts, q=q, ua=ua, da=da, up=up,
                         eu=q - ua - fee(ua), ed=(1 - q) - da - fee(da),
                         mid=(r.up_bid[0] + ua) / 2 if r.up_bid else ua))
df = pd.DataFrame(rows)
print(f"windows with outcome: {df.ts.nunique()}  snapshots: {len(df)}")
print(f"brier model={np.mean((df.q - df.up) ** 2):.4f}  market_mid={np.mean((df.mid - df.up) ** 2):.4f}")
for lo, hi in [(0, 120), (120, 240), (240, 270), (270, 295)]:
    d = df[(df.el >= lo) & (df.el < hi)]
    print(f"  el {lo:3d}-{hi:3d}s brier model={np.mean((d.q-d.up)**2):.4f} mkt={np.mean((d.mid-d.up)**2):.4f}")

for thr in THR:
    trades = []
    for ts, g in df.groupby("ts"):
        hit = g[(g.eu > thr) | (g.ed > thr)]
        if hit.empty:
            continue
        r = hit.iloc[0]
        if r.eu >= r.ed:
            trades.append(((r.up - r.ua - fee(r.ua)) / r.ua, r.el))
        else:
            trades.append((((not r.up) - r.da - fee(r.da)) / r.da, r.el))
    if trades:
        a = np.array([x[0] for x in trades])
        print(f"thr={thr:.2f} trades={len(a):3d} roi/trade={a.mean():+.4f} "
              f"t={a.mean() / (a.std() + 1e-9) * sqrt(len(a)):+.2f} win={np.mean(a > 0):.3f} "
              f"median_entry_s={np.median([x[1] for x in trades]):.0f}")
