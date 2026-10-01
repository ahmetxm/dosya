"""Outcome-only hypotheses: streaks, hour-of-day, prior-window momentum. Bets placed at the first recorded price after open."""
import numpy as np
import pandas as pd
from load import load, fee

mk, btc = load()
rows = []
for m in mk:
    p0 = next((h["p"] for h in m["hist"] if h["t"] >= m["ts"]), None)
    rows.append(dict(ts=m["ts"], up=float(m["up_won"] > .5), p0=p0))
df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
df = df[df.ts.diff().fillna(300) == 300].reset_index(drop=True)
df["prev"] = df.up.shift(1)
df["prev2"] = df.up.shift(2)
df["prev3"] = df.up.shift(3)
df["hour"] = pd.to_datetime(df.ts, unit="s").dt.hour
df = df.dropna()


def bet(mask, side_up, label):
    d = df[mask]
    if not len(d):
        return
    price = np.where(side_up, d.p0 + 0.01, 1 - d.p0 + 0.01)
    win = d.up if side_up else 1 - d.up
    roi = (win - price - fee(price)) / price
    print(f"{label:40s} n={len(d):5d} hit={win.mean():.3f} roi={roi.mean():+.4f} "
          f"t={roi.mean()/roi.std()*np.sqrt(len(d)):+.2f}")


print("base up rate", df.up.mean().round(4), " lag-1 autocorr", df.up.autocorr(1).round(4))
bet(df.prev == 1, True, "after Up -> buy Up (momentum)")
bet(df.prev == 1, False, "after Up -> buy Down (reversal)")
bet((df.prev == 1) & (df.prev2 == 1) & (df.prev3 == 1), False, "after 3 Up -> buy Down")
bet((df.prev == 0) & (df.prev2 == 0) & (df.prev3 == 0), True, "after 3 Down -> buy Up")
bet(df.p0 < 0.45, True, "open price <0.45 -> buy Up (fade)")
bet(df.p0 > 0.55, False, "open price >0.55 -> buy Down (fade)")
print("\nhour-of-day Up rate (UTC), |t|>2 flagged:")
for h, g in df.groupby("hour"):
    t = (g.up.mean() - .5) / (0.5 / np.sqrt(len(g)))
    print(f"  {h:02d}h n={len(g)} up={g.up.mean():.3f} t={t:+.2f}{' *' if abs(t) > 2 else ''}")
