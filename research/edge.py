"""Compare a simple TWAP-aware model against Polymarket prices at several decision times."""
import numpy as np
import pandas as pd
import sys
from load import load, fee, price_at
LAG=int(sys.argv[1]) if len(sys.argv)>1 else 0

def price_after(hist,t):
    for h in hist:
        if h["t"]>=t: return h["p"], h["t"]-t
    return None, None


mk, btc = load()
ret = np.log(btc.close).diff()
vol = ret.rolling(60).std()

rows = []
for m in mk:
    ts = m["ts"]
    if ts - 60 not in btc.index or ts + 240 not in btc.index:
        continue
    ref = btc.loc[ts - 60, "mid"]
    for off in [0, 60, 120, 180, 240]:
        # Decision at ts+off using only candles fully closed by then
        last = ts + off - 60
        spot = btc.loc[last, "close"]
        sig = vol.get(last, np.nan)
        p, dt = price_after(m["hist"], ts + off + LAG)
        if p is None or dt > 75 or np.isnan(sig):
            continue
        if off == 240:
            partial = 0.0
        rows.append(dict(ts=ts, off=off, z=np.log(spot / ref) / sig, p=p,
                         up=float(m["up_won"] > 0.5)))
df = pd.DataFrame(rows)
df["day"] = (df.ts // 86400)
days = sorted(df.day.unique())
train = df[df.day.isin(days[: len(days) // 2])]
test = df[df.day.isin(days[len(days) // 2:])]


def fit_logit(x, y, iters=200):
    X = np.c_[np.ones_like(x), x]
    w = np.zeros(2)
    for _ in range(iters):
        q = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - q)
        H = (X * (q * (1 - q))[:, None]).T @ X + 1e-6 * np.eye(2)
        w += np.linalg.solve(H, g)
    return w


print("off  n_test  brier_model brier_mkt   acc_model acc_mkt")
for off in sorted(df.off.unique()):
    tr, te = train[train.off == off], test[test.off == off]
    w = fit_logit(tr.z.values, tr.up.values)
    q = 1 / (1 + np.exp(-(w[0] + w[1] * te.z.values)))
    bm = np.mean((q - te.up) ** 2)
    bk = np.mean((te.p - te.up) ** 2)
    print(f"{off:3d} {len(te):6d}  {bm:.4f}      {bk:.4f}     {np.mean((q>.5)==te.up):.3f}    {np.mean((te.p>.5)==te.up):.3f}")
    test.loc[te.index, "q"] = q

print("\nTaker strategy on test half: buy side where model prob - price - fee > thr (assumes fill at recorded price + 1c)")
for off in sorted(df.off.unique()):
    te = test[test.off == off]
    for thr in [0.0, 0.03, 0.06, 0.1]:
        pnl = []
        for _, r in te.iterrows():
            pu = min(r.p + 0.01, 0.99)
            pd_ = min(1 - r.p + 0.01, 0.99)
            eu = r.q - pu - fee(pu)
            ed = (1 - r.q) - pd_ - fee(pd_)
            if eu > thr and eu >= ed:
                pnl.append((r.up - pu - fee(pu)) / pu)
            elif ed > thr:
                pnl.append(((1 - r.up) - pd_ - fee(pd_)) / pd_)
        if pnl:
            a = np.array(pnl)
            print(f"off={off:3d} thr={thr:.2f} trades={len(a):4d} roi/trade={a.mean():+.4f} "
                  f"t={a.mean()/a.std()*np.sqrt(len(a)):+.2f} win={np.mean(a>0):.3f}")
