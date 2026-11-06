import sys, numpy as np, pandas as pd
cols = pd.read_csv(sys.argv[1]); lab = pd.read_csv(sys.argv[2], header=None, names=["id", "flux", "rejection"])
h = lab[lab.rejection > 0.9].flux.values
cnt, edges = np.histogram(h, bins=30)
best = None
for x0 in np.arange(150, 160, 0.25):
    for s in np.arange(5.40, 5.60, 0.005):
        f = (cols.x.values - x0) / s
        idx = np.searchsorted(edges, f, side="right") - 1
        ok = (idx >= 0) & (idx < 30) & (np.abs(f - edges[np.clip(idx,0,29)]) > 0.3) & (np.abs(f - edges[np.clip(idx+1,0,30)]) > 0.3)
        pred = np.where((idx >= 0) & (idx < 30), cnt[np.clip(idx, 0, 29)], 0)
        k = (cols.data.values[ok] @ pred[ok]) / (pred[ok] @ pred[ok])
        err = np.abs(cols.data.values[ok] - k * pred[ok]).mean()
        if best is None or err < best[0]:
            best = (err, x0, s, k)
print("err px %.2f  x0 %.2f  px/flux %.3f  px/count %.4f" % best)
err, x0, s, k = best
cols["flux"] = (cols.x - x0) / s
for key in ("drl", "llm"):
    v = cols[key].values
    # split into constant segments (bars)
    seg, start = [], 0
    for i in range(1, len(v) + 1):
        if i == len(v) or abs(int(v[i]) - int(v[start])) > 2:
            seg.append((cols.flux.iloc[start], cols.flux.iloc[i - 1], np.median(v[start:i])))
            start = i
    seg = [(a, b, hgt) for a, b, hgt in seg if b - a > 1.5]
    widths = [b - a for a, b, _ in seg if b - a > 3]
    tot = sum(hgt / k for _, _, hgt in seg)
    print(key, "bars", len(seg), "median width %.2f" % np.median(widths), "total count %.0f" % tot)
    pd.DataFrame(seg, columns=["flux_lo", "flux_hi", "height_px"]).assign(count=lambda d: d.height_px / k).to_csv(sys.argv[3] + f"_{key}.csv", index=False)
