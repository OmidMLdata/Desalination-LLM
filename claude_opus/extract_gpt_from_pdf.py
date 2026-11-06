"""Recover GPT-6.1 per-step series from the vector matplotlib PDF (six_category_comparison.pdf)."""
import sys, numpy as np, pandas as pd, pdfplumber
pdf = sys.argv[1]; out = sys.argv[2]
pg = pdfplumber.open(pdf).pages[0]
FX = lambda x: (x - 46.61) / ((314.24 - 46.61) / 30)
FY = lambda y: (209.16 - y) * 50 / (209.16 - 161.82)
RX = lambda x: (x - 384.57) / ((652.19 - 384.57) / 30)
RY = lambda y: 0.5 + (209.82 - y) * 0.5 / (209.82 - 25.04)
def key(c):
    s = c.get("stroking_color"); s = tuple(round(v, 3) for v in s) if isinstance(s, (tuple, list)) else round(s, 3)
    return s, bool(c.get("dash") and c["dash"][0])
names = {((0.0, 0.447, 0.698), False): "gpt_tools", ((0.835, 0.369, 0.0), False): "gpt_notools",
         ((0.835, 0.369, 0.0), True): "gpt_notools_claimed", ((0.0, 0.62, 0.451), False): "scaffold_greedy",
         ((0.902, 0.624, 0.0), True): "scaffold_constrained", (0.467, True): "scaffold_random"}
series, bands = {}, {}
for c in pg.curves:
    pts = np.array(c["pts"]); left = pts[:, 0].min() < 340
    k = key(c)
    if c.get("fill") and len(pts) == 64:   # fill_between polygon: 31 upper + 31 lower + closure
        nm = {(0.0, 0.62, 0.451): "scaffold_greedy", (0.902, 0.624, 0.0): "scaffold_constrained", 0.467: "scaffold_random"}[k[0]]
        fx, fy = (FX, FY) if left else (RX, RY)
        p = pts[1:63] if len(pts) == 64 else pts
        st = np.rint(fx(p[:, 0])).astype(int); val = fy(p[:, 1])
        df = pd.DataFrame({"step": st, "v": val}).groupby("step").v.agg(["min", "max"])
        bands[(nm, "flux" if left else "rejection")] = df
    elif len(pts) == 31 and k in names:
        fx, fy = (FX, FY) if left else (RX, RY)
        series[(names[k], "flux" if left else "rejection")] = pd.Series(fy(pts[:, 1]), index=np.rint(fx(pts[:, 0])).astype(int))
rows = []
for nm in names.values():
    for s in range(31):
        r = {"condition": nm, "step": s, "flux": series[(nm, "flux")][s], "rejection": series[(nm, "rejection")][s]}
        for m in ("flux", "rejection"):
            if (nm, m) in bands:
                b = bands[(nm, m)]; r[f"{m}_lo"], r[f"{m}_hi"] = b.loc[s, "min"], b.loc[s, "max"]
        rows.append(r)
d = pd.DataFrame(rows); d.to_csv(out, index=False)
print(d.groupby("condition").apply(lambda g: g.iloc[[1, 10, 20, 30]][["step", "flux", "rejection"]].round(3).values.tolist()).to_string())
