"""Digitise the DRL / dataset / LLM histograms of the original manuscript Figure 4b
(raster panel embedded in figs/Figure4.pdf). Heights -> counts with the dataset bars,
whose true counts are known from Graphene-RL data/noisy_augmented_label.csv."""
import sys, numpy as np, pandas as pd
from PIL import Image
rgb, mask, out = sys.argv[1], sys.argv[2], sys.argv[3]
im = np.array(Image.open(rgb).convert("RGB")).astype(float); a = np.array(Image.open(mask)).astype(float)[..., None] / 255
c = im * a + 255 * (1 - a)
REF = {"W": (255, 255, 255), "B": (186, 212, 232), "T": (214, 191, 168), "P": (255, 216, 182),
       "LW": (151, 103, 93), "LP": (182, 110, 69), "LT": (157, 114, 88), "LB": (139, 111, 100)}
names = list(REF); refs = np.array([REF[k] for k in names])
d = ((c[..., None, :] - refs[None, None]) ** 2).sum(-1)
lab = np.array(names)[d.argmin(-1)]
grid_like = (np.abs(c[..., 0] - c[..., 1]) < 8) & (np.abs(c[..., 1] - c[..., 2]) < 8) & (c[..., 0] < 245)
lab[grid_like] = "G"
BASE, X0, X1, YTOP = 711, 156, 1117, 30
has = {"drl": {"B", "T", "LT", "LB"}, "data": {"P", "T", "LP", "LT"}, "llm": {"LW", "LP", "LT", "LB"}}
rows = []
for x in range(X0, X1 + 1):
    col = lab[:, x]
    r = {"x": x}
    for k, s in has.items():
        ylo = 175 if 805 <= x <= 1105 else YTOP          # skip the legend box
        ys = [y for y in range(ylo, BASE) if col[y] in s]
        # require a run of >=3 pixels to ignore anti-aliasing
        top = BASE
        for y in ys:
            if all(col[y + i] in s or col[y + i] == "G" for i in range(3)):
                top = y; break
        r[k] = BASE - top
    rows.append(r)
df = pd.DataFrame(rows); df.to_csv(out, index=False)
print(df.describe().round(1))
