import numpy as np
from load import load

mk, btc = load()
res = {k: [] for k in ["close_vs_open", "twap5_vs_open", "lastmin_vs_prevmin", "twap5_vs_prevmin"]}
for m in mk:
    ts = m["ts"]
    mins = [ts + 60 * i for i in range(5)]
    if not all(t in btc.index for t in mins + [ts - 60]):
        continue
    w = btc.loc[mins]
    o = btc.loc[ts, "open"]
    up = m["up_won"] > 0.5
    res["close_vs_open"].append((w.close.iloc[-1] >= o) == up)
    res["twap5_vs_open"].append((w.mid.mean() >= o) == up)
    res["lastmin_vs_prevmin"].append((w.mid.iloc[-1] >= btc.loc[ts - 60, "mid"]) == up)
    res["twap5_vs_prevmin"].append((w.mid.mean() >= btc.loc[ts - 60, "mid"]) == up)
for k, v in res.items():
    print(f"{k:22s} n={len(v)} agree={np.mean(v):.4f}")
