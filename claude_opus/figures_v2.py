"""Manuscript figures for the Claude Opus 5.5 vs GPT-6.1 Sol benchmark (GPT-6.1 protocol:
30 steps, up to 30 trials/step, <=3 atoms/trial with colonisation, 90 % rejection floor).

  python -m claude_opus.figures_v2 --out <figs dir>

Writes Figure3.pdf, Figure4.pdf, Figure5.pdf, FigureSI1.pdf, Figure2SI.pdf, FigureSI_calibration.pdf,
and table_metrics.csv / table_metrics.tex.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch

from desalination_llm.geometry import FEATURE_NAMES
from desalination_llm.pdb import GrapheneStructure

from .env import REPO

R = REPO / "results" / "claude_opus"
GPT = REPO / "data" / "external" / "gpt61_six_category_extracted.csv"
FLOOR = 0.90
SEEDS = [42, 1, 2, 3, 4, 5]

# Claude template: one colour per category, same for both models
C = {"tools": "#2a78d6", "notools": "#eb6834", "greedy": "#1baf7a", "constrained": "#eda100", "random": "#8a8983"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LBL = {"tools": "{m} + tools (agent)", "notools": "{m} no tools, predictor score",
       "claimed": "{m} no tools, own estimate", "greedy": "Scaffold only, greedy (no LLM)",
       "constrained": "Scaffold only, greedy + manufacturability rules", "random": "Scaffold only, random feasible"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "pdf.fonttype": 42, "savefig.bbox": "tight"})


# ----------------------------------------------------------------------------- data
def claude_steps():
    d = {"tools": pd.read_csv(R / "opus_tools_per_step.csv"), "notools": pd.read_csv(R / "opus_notools_per_step.csv")}
    cl = d["notools"].copy()
    cl.loc[1:, "flux"], cl.loc[1:, "rejection"] = cl.loc[1:, "claimed_flux"], cl.loc[1:, "claimed_rejection"]
    d["claimed"] = cl
    for p in ("greedy", "constrained", "random"):
        runs = [pd.read_csv(R / f"scaffold_{p}_s{s}_per_step.csv") for s in SEEDS if (R / f"scaffold_{p}_s{s}_per_step.csv").exists()]
        d[p + "_runs"] = runs
        st = pd.concat([r.set_index("step")[["flux", "rejection"]] for r in runs], keys=range(len(runs)))
        g = st.groupby(level=1)
        d[p] = pd.DataFrame({"step": g.mean().index, "flux": g.mean().flux.values, "rejection": g.mean().rejection.values,
                             "flux_sd": g.std().flux.fillna(0).values, "rejection_sd": g.std().rejection.fillna(0).values})
    return d


def gpt_steps():
    g = pd.read_csv(GPT)
    m = {"gpt_tools": "tools", "gpt_notools": "notools", "gpt_notools_claimed": "claimed",
         "scaffold_greedy": "greedy", "scaffold_constrained": "constrained", "scaffold_random": "random"}
    out = {}
    for k, v in m.items():
        x = g[g.condition == k].sort_values("step").reset_index(drop=True)
        if "flux_lo" in x and x.flux_lo.notna().any():
            x["flux_sd"] = (x.flux_hi - x.flux_lo) / 2; x["rejection_sd"] = (x.rejection_hi - x.rejection_lo) / 2
        out[v] = x
    return out


def claude_trials():
    fr = [pd.read_csv(R / "opus_tools_trials.csv").assign(cond="tools", seed=42)]
    for p in ("greedy", "constrained", "random"):
        for s in SEEDS:
            f = R / f"scaffold_{p}_s{s}_trials.csv"
            if f.exists():
                fr.append(pd.read_csv(f).assign(cond=p, seed=s))
    return pd.concat(fr, ignore_index=True)


def valid(df):
    return (df.rejection >= FLOOR) & (df.connected_components <= 1) & (df.internal_unremoved_count == 0)


# ----------------------------------------------------------------------------- Figure 3
def _panel(ax, d, metric, model, show_claimed=True):
    for p, ls, lw in (("random", ":", 1.6), ("greedy", "-", 1.5), ("constrained", (0, (6, 2.6)), 1.8)):
        x = d[p]
        if f"{metric}_sd" in x:
            ax.fill_between(x.step, x[metric] - x[f"{metric}_sd"], x[metric] + x[f"{metric}_sd"], color=C[p], alpha=0.13, lw=0)
        ax.plot(x.step, x[metric], ls=ls, color=C[p], lw=lw)
    ax.plot(d["notools"].step, d["notools"][metric], color=C["notools"], lw=1.6)
    if show_claimed:
        ax.plot(d["claimed"].step, d["claimed"][metric], color=C["notools"], lw=1.5, ls=(0, (5, 2.3)))
    ax.plot(d["tools"].step, d["tools"][metric], color=C["tools"], lw=2.0)
    if metric == "rejection":
        ax.axhline(FLOOR, color=INK2, lw=0.9, ls="--")
        ax.text(30, FLOOR - 0.022, "90 % rejection floor", fontsize=7.5, color=INK2, ha="right")
        ax.set_ylim(0.5, 1.01); ax.set_ylabel("Ion rejection")
    else:
        ax.set_ylabel("Water flux (ns$^{-1}$)"); ax.set_ylim(bottom=-3)
    ax.set_xlim(-0.5, 30.5); ax.set_xlabel("Design step")


def legend_handles(model):
    return [Line2D([], [], color=C["tools"], lw=2.0, label=LBL["tools"].format(m=model)),
            Line2D([], [], color=C["notools"], lw=1.6, label=LBL["notools"].format(m=model)),
            Line2D([], [], color=C["notools"], lw=1.5, ls=(0, (5, 2.3)), label=LBL["claimed"].format(m=model)),
            Line2D([], [], color=C["greedy"], lw=1.5, label=LBL["greedy"]),
            Line2D([], [], color=C["constrained"], lw=1.8, ls=(0, (6, 2.6)), label=LBL["constrained"]),
            Line2D([], [], color=C["random"], lw=1.6, ls=":", label=LBL["random"])]


def figure3(cd, gd, out):
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.6))
    _panel(ax[0, 0], cd, "flux", "Claude Opus 5.5"); _panel(ax[0, 1], cd, "rejection", "Claude Opus 5.5")
    _panel(ax[1, 0], gd, "flux", "GPT-6.1 Sol"); _panel(ax[1, 1], gd, "rejection", "GPT-6.1 Sol")
    for a, t in zip(ax.flat, ["(a) Claude Opus 5.5: water flux", "(b) Claude Opus 5.5: ion rejection",
                              "(c) GPT-6.1 Sol: water flux", "(d) GPT-6.1 Sol: ion rejection"]):
        a.set_title(t, loc="left", fontsize=10, fontweight="bold", color=INK)
    h = legend_handles("LLM")
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=8.2, bbox_to_anchor=(0.5, -0.045))
    fig.text(0.5, -0.075, "Both models: 30 steps, up to 30 trials/step, <=3 atoms/trial (colonisation included), 90 % floor; "
             "scaffold lines = mean of 6 seeds, bands = +/-1 SD", ha="center", fontsize=7.5, color=INK2)
    fig.tight_layout(h_pad=2.2)
    fig.savefig(out / "Figure3.pdf"); fig.savefig(out / "Figure3.png", dpi=200); plt.close(fig)


# ----------------------------------------------------------------------------- pore thumbnails
_ST = GrapheneStructure.from_pdb(REPO / "data/structures/pristine_graphene_4040.pdb")
_XY, _S2I = _ST.xy, _ST.serial_to_index()


def pore_image(removed_serials: str, half=7.5, px=160):
    rem = [_S2I[int(s)] for s in str(removed_serials).split()] if isinstance(removed_serials, str) and removed_serials.strip() else []
    c = _XY[rem].mean(0) if rem else _XY.mean(0)
    keep = np.setdiff1d(np.arange(len(_XY)), rem)
    f, a = plt.subplots(figsize=(1.2, 1.2), dpi=px / 1.2)
    a.scatter(_XY[keep, 0], _XY[keep, 1], s=22, c="#5ec4c4", edgecolor="#2b8c8c", lw=0.4)
    a.set_xlim(c[0] - half, c[0] + half); a.set_ylim(c[1] - half, c[1] + half); a.set_aspect("equal"); a.axis("off")
    f.subplots_adjust(0, 0, 1, 1)
    buf = io.BytesIO(); f.savefig(buf, format="png", dpi=px / 1.2, facecolor="white"); plt.close(f); buf.seek(0)
    return plt.imread(buf)


def add_thumb(ax, xy, img, box, text, zoom=0.30, coords="axes fraction"):
    ab = AnnotationBbox(OffsetImage(img, zoom=zoom), xy, xybox=box, boxcoords=coords, frameon=True, pad=0.15,
                        bboxprops=dict(edgecolor=INK2, lw=0.6), arrowprops=dict(arrowstyle="-", ls="--", color="#3b1e2e", lw=0.8))
    ax.add_artist(ab)
    below = box[1] >= 0.5
    tb = (box[0], box[1] - 0.115) if below else (box[0], box[1] + 0.115)
    ax.annotate(text, xy=tb, xycoords=coords, ha="center", va="top" if below else "bottom",
                fontsize=7, color=INK, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


# ----------------------------------------------------------------------------- Figure 4
def figure4(cd, gd, tr, out):
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.05, 1]})
    a = ax[0]
    for p in ("random", "greedy", "constrained"):
        s = tr[tr.cond == p]
        a.scatter(s.flux, s.rejection, s=6, color=C[p], alpha=0.25, lw=0, label=f"{LBL[p]} trials ({len(s)})", rasterized=True)
    s = tr[tr.cond == "tools"]
    a.scatter(s.flux, s.rejection, s=9, color=C["tools"], alpha=0.55, lw=0, label=f"Claude Opus 5.5 + tools trials ({len(s)})", rasterized=True)
    t = cd["tools"]; n = cd["notools"]
    a.plot(t.flux, t.rejection, "-o", color=C["tools"], ms=3.2, lw=1.4, mec="white", mew=0.5, label="Claude Opus 5.5 + tools, selected path")
    a.plot(n.flux, n.rejection, "-o", color=C["notools"], ms=3.2, lw=1.4, mec="white", mew=0.5, label="Claude Opus 5.5 no tools, selected path")
    a.axhline(FLOOR, color=INK2, lw=0.9, ls="--")
    # best valid agent design
    v = t[valid(t)]; b = v.loc[v.flux.idxmax()]
    add_thumb(a, (b.flux, b.rejection), pore_image(b.removed_serials), (0.45, 0.50),
              f"Opus 5.5 + tools, best valid design\n{b.flux:.1f} ns$^{{-1}}$, {b.rejection:.1%}, {int(b.removed_count)} C removed", zoom=0.30)
    a.set_xlabel("Water flux (ns$^{-1}$)"); a.set_ylabel("Ion rejection"); a.set_ylim(0.6, 1.0); a.set_xlim(left=-3)
    a.legend(fontsize=6.6, loc="lower left", markerscale=1.6)
    a.set_title("(a) Performance window, Claude Opus 5.5 runs", loc="left", fontsize=10, fontweight="bold")

    # (b) histogram of flux for designs with rejection >= 90 %
    a = ax[1]
    hi = tr[tr.rejection >= FLOOR]
    bins = np.arange(0, 180, 7.5)
    ag = hi[hi.cond == "tools"].flux; sc = hi[hi.cond != "tools"].flux
    gt = gd["tools"]; gt = gt[(gt.rejection >= FLOOR) & (gt.step > 0)]
    gsel = gt.loc[~gt.flux.round(2).duplicated()].flux      # distinct designs (keep-steps repeat the pore)
    a.hist(sc, bins=bins, color=C["constrained"], alpha=0.55, label=f"Scaffold only, all policies x 6 seeds: {len(sc)} pores")
    a.hist(ag, bins=bins, color=C["tools"], alpha=0.75, label=f"Claude Opus 5.5 + tools: {len(ag)} pores")
    a2 = a.twinx(); a2.grid(False); a2.spines["right"].set_visible(True)
    a2.hist(gsel, bins=bins, histtype="step", color="#4a3aa7", lw=1.6, label=f"GPT-6.1 Sol + tools, distinct selected designs: {len(gsel)} (right axis)")
    a2.set_ylabel("GPT-6.1 Sol selected designs", color="#4a3aa7"); a2.tick_params(axis="y", colors="#4a3aa7")
    a.set_xlabel("Water flux (ns$^{-1}$)"); a.set_ylabel("No. pores with rejection $\\geq$ 90 %")
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    a.set_title("(b) Flux distribution of high-rejection pores", loc="left", fontsize=10, fontweight="bold")
    # inset: top window
    lo = 100
    ins = a.inset_axes([0.50, 0.30, 0.36, 0.34])
    bi = np.arange(lo, 172.5, 5)
    ins.hist(sc[sc >= lo], bins=bi, color=C["constrained"], alpha=0.55)
    ins.hist(ag[ag >= lo], bins=bi, color=C["tools"], alpha=0.8)
    vv = hi[valid(hi) & (hi.flux >= lo)]
    ins.set_title(f"flux $\\geq$ {lo} ns$^{{-1}}$: {len(hi[hi.flux >= lo])} Claude-run pores\n({len(vv)} contiguous and island-free)", fontsize=6.5)
    ins.tick_params(labelsize=6.5); ins.grid(alpha=0.4)
    fig.tight_layout()
    fig.savefig(out / "Figure4.pdf", dpi=300); fig.savefig(out / "Figure4.png", dpi=200); plt.close(fig)


# ----------------------------------------------------------------------------- Figure 5 (t-SNE)
def figure5(cd, tr, out):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    nt = cd["notools"].iloc[1:].copy()
    feats = [f"f_{k}" for k in FEATURE_NAMES]
    # geometric descriptors of the no-tool designs (computed from their removed atoms)
    from desalination_llm.geometry import build_bond_graph, geometry_features
    adj, bp = build_bond_graph(_XY)
    for k in FEATURE_NAMES:
        nt[f"f_{k}"] = [geometry_features(_XY, [_S2I[int(s)] for s in str(r).split()], adj, bp)[k] for r in nt.removed_serials]
    nt["cond"] = "notools"
    data = pd.concat([tr, nt], ignore_index=True)
    data = data.drop_duplicates(subset=["removed_serials", "cond"]).reset_index(drop=True)
    X = StandardScaler().fit_transform(data[feats].fillna(0).to_numpy())
    Y = TSNE(n_components=2, perplexity=40, init="pca", random_state=0, max_iter=1500).fit_transform(X)
    data["t1"], data["t2"] = Y[:, 0], Y[:, 1]
    data.to_csv(R / "tsne_embedding.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.0))
    for a, col, lab, cm, fmt in [(ax[0], "flux", "Water flux (ns$^{-1}$)", "cividis", "{:.1f}"),
                                 (ax[1], "rejection", "Ion rejection", "cividis", "{:.1%}")]:
        sc = a.scatter(data.t1, data.t2, c=data[col], cmap=cm, s=9, lw=0, alpha=0.85, rasterized=True)
        cb = fig.colorbar(sc, ax=a, fraction=0.04, pad=0.02); cb.set_label(lab)
        if col == "rejection":
            cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        # agent paths
        for cond, c in (("tools", C["tools"]), ("notools", C["notools"])):
            sel = cd[cond].iloc[1:]
            pts = data[(data.cond == cond) & data.removed_serials.isin(sel.removed_serials)]
            pts = pts.drop_duplicates("removed_serials").set_index("removed_serials").reindex(sel.removed_serials.drop_duplicates()).dropna(subset=["t1"])
            a.plot(pts.t1, pts.t2, "-o", color=c, ms=4, lw=1.1, mec="white", mew=0.6, zorder=4)
        # thumbnails at characteristic designs
        vs = data[(data.cond == "constrained") & valid(data)]
        picks = [data[data.f_removed_atoms_count > 0].sort_values("f_removed_atoms_count").iloc[0],
                 data[(data.cond == "tools") & valid(data)].sort_values("flux").iloc[-1],
                 data[data.cond == "notools"].sort_values("step").iloc[-1],
                 data.loc[data.flux.idxmax()],
                 vs.loc[vs.flux.idxmax()],
                 data.loc[(data.flux - data.flux.median()).abs().idxmin()]]
        # place each thumbnail on the axes border nearest to its point
        slots = [(0.09, 0.86), (0.36, 0.88), (0.64, 0.88), (0.91, 0.86), (0.09, 0.12), (0.36, 0.10), (0.64, 0.10), (0.91, 0.12)]
        xr = (data.t1.min(), data.t1.max()); yr = (data.t2.min(), data.t2.max())
        used = set()
        for r in picks:
            fx = (r.t1 - xr[0]) / (xr[1] - xr[0]); fy = (r.t2 - yr[0]) / (yr[1] - yr[0])
            j = min((i for i in range(len(slots)) if i not in used), key=lambda i: (slots[i][0] - fx) ** 2 + (slots[i][1] - fy) ** 2)
            used.add(j)
            add_thumb(a, (r.t1, r.t2), pore_image(r.removed_serials), slots[j], fmt.format(r[col]), zoom=0.27)
        a.set_xlabel("t-SNE dimension 1"); a.set_ylabel("t-SNE dimension 2"); a.grid(False)
        a.set_xticks([]); a.set_yticks([])
        pad = 0.30
        a.set_xlim(xr[0] - pad * (xr[1] - xr[0]) * 0.5, xr[1] + pad * (xr[1] - xr[0]) * 0.5)
        a.set_ylim(yr[0] - pad * (yr[1] - yr[0]), yr[1] + pad * (yr[1] - yr[0]))
    ax[0].set_title("(a) coloured by predicted water flux", loc="left", fontsize=10, fontweight="bold", pad=10)
    ax[1].set_title("(b) coloured by predicted ion rejection", loc="left", fontsize=10, fontweight="bold", pad=10)
    h = [Line2D([], [], color=C["tools"], marker="o", lw=1.1, label="Claude Opus 5.5 + tools, selected path"),
         Line2D([], [], color=C["notools"], marker="o", lw=1.1, label="Claude Opus 5.5 no tools, selected path")]
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    fig.savefig(out / "Figure5.pdf", dpi=300); fig.savefig(out / "Figure5.png", dpi=200); plt.close(fig)
    return data


# ----------------------------------------------------------------------------- SI figures
def figure_si1(cd, tr, out):
    fig = plt.figure(figsize=(12, 6.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.25, 1])
    a = fig.add_subplot(gs[0, :])
    for p in ("random", "greedy", "constrained"):
        for s in SEEDS:
            x = tr[(tr.cond == p) & (tr.seed == s)]
            a.scatter(x.flux, x.rejection, s=5, color=C[p], alpha=0.22, lw=0, rasterized=True)
    x = tr[tr.cond == "tools"]
    a.scatter(x.flux, x.rejection, s=8, color=C["tools"], alpha=0.6, lw=0, rasterized=True)
    a.axhline(FLOOR, color=INK2, lw=0.9, ls="--")
    a.set_xlabel("Water flux (ns$^{-1}$)"); a.set_ylabel("Ion rejection")
    a.legend(handles=[Line2D([], [], marker="o", ls="", color=C[k], label=l) for k, l in
                      [("tools", "Claude Opus 5.5 + tools"), ("greedy", LBL["greedy"]), ("constrained", LBL["constrained"]), ("random", LBL["random"])]],
             fontsize=7.5, loc="lower left")
    a.set_title("(a) All evaluated trial pores (Claude Opus 5.5 run and 18 scaffold runs)", loc="left", fontsize=10, fontweight="bold")
    final = [("Opus 5.5 + tools", cd["tools"].iloc[-1], C["tools"]), ("Opus 5.5 no tools", cd["notools"].iloc[-1], C["notools"])]
    for p in ("greedy", "constrained", "random"):
        final.append((LBL[p].replace("Scaffold only, ", "Scaffold: ") + " (seed 42)", pd.read_csv(R / f"scaffold_{p}_s42_per_step.csv").iloc[-1], C[p]))
    for i, (name, r, c) in enumerate(final):
        b = fig.add_subplot(gs[1, i])
        rem = [_S2I[int(s)] for s in str(r.removed_serials).split()] if isinstance(r.removed_serials, str) else []
        cen = _XY[rem].mean(0) if rem else _XY.mean(0)
        keep = np.setdiff1d(np.arange(len(_XY)), rem)
        b.scatter(_XY[keep, 0], _XY[keep, 1], s=11, c="#5ec4c4", edgecolor="#2b8c8c", lw=0.3)
        b.scatter(_XY[rem, 0], _XY[rem, 1], s=16, facecolor="none", edgecolor=c, lw=1.0)
        b.set_xlim(cen[0] - 10, cen[0] + 10); b.set_ylim(cen[1] - 10, cen[1] + 10); b.set_aspect("equal"); b.axis("off")
        ok = "valid" if (r.rejection >= FLOOR and r.connected_components <= 1 and r.internal_unremoved_count == 0) else "invalid"
        b.set_title(f"{name}\nflux {r.flux:.1f} ns$^{{-1}}$, rej. {r.rejection:.1%}\n{int(r.removed_count)} C removed, "
                    f"{int(r.connected_components)} cluster(s), {ok}", fontsize=7)
    fig.text(0.01, 0.40, "(b)", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / "FigureSI1.pdf", dpi=300); fig.savefig(out / "FigureSI1.png", dpi=200); plt.close(fig)


def figure_si2(cd, out):
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for p, dx in (("greedy", -0.22), ("constrained", 0.0), ("random", 0.22)):
        x = cd[p]
        for a, m in zip(ax, ("flux", "rejection")):
            a.errorbar(x.step + dx, x[m], yerr=x[f"{m}_sd"], fmt="o", ms=3, color=C[p], ecolor=C[p], elinewidth=0.9, capsize=1.5, label=LBL[p])
    ax[0].set_ylabel("Water flux (ns$^{-1}$)"); ax[1].set_ylabel("Ion rejection"); ax[1].axhline(FLOOR, color=INK2, lw=0.9, ls="--")
    for a, t in zip(ax, ("(a) Water flux, mean +/- SD (6 seeds)", "(b) Ion rejection, mean +/- SD (6 seeds)")):
        a.set_xlabel("Design step"); a.set_title(t, loc="left", fontsize=10, fontweight="bold")
    ax[0].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "Figure2SI.pdf"); fig.savefig(out / "Figure2SI.png", dpi=200); plt.close(fig)


def figure_si_cal(cd, gd, out):
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for d, m, mk in ((cd, "Claude Opus 5.5", "o"), (gd, "GPT-6.1 Sol", "s")):
        p, c = d["notools"].iloc[1:], d["claimed"].iloc[1:]
        for a, k in zip(ax, ("flux", "rejection")):
            a.scatter(p[k], c[k], s=24, marker=mk, color=C["notools"] if mk == "o" else "#4a3aa7", edgecolor="white", lw=0.6, label=m)
    for a, k, lab in zip(ax, ("flux", "rejection"), ("water flux (ns$^{-1}$)", "ion rejection")):
        lim = [min(a.get_xlim()[0], a.get_ylim()[0]), max(a.get_xlim()[1], a.get_ylim()[1])]
        a.plot(lim, lim, color=INK2, lw=0.9); a.set_xlim(lim); a.set_ylim(lim)
        a.set_xlabel(f"Predictor {lab}"); a.set_ylabel(f"Model's own estimate of {lab}")
    ax[1].axhline(FLOOR, color=INK2, ls="--", lw=0.8); ax[1].axvline(FLOOR, color=INK2, ls="--", lw=0.8)
    ax[0].legend(fontsize=8)
    ax[0].set_title("(a) Flux: own estimate vs predictor", loc="left", fontsize=10, fontweight="bold")
    ax[1].set_title("(b) Rejection: own estimate vs predictor", loc="left", fontsize=10, fontweight="bold")
    fig.tight_layout(); fig.savefig(out / "FigureSI_calibration.pdf"); fig.savefig(out / "FigureSI_calibration.png", dpi=200); plt.close(fig)


# ----------------------------------------------------------------------------- metrics table
def metrics(cd, gd):
    rows = []
    for model, d, geom in (("Claude Opus 5.5", cd, True), ("GPT-6.1 Sol", gd, False)):
        ref = d["constrained"].iloc[-1].flux
        for k in ("tools", "notools", "greedy", "constrained", "random"):
            x = d[k]; last = x.iloc[-1]; s = x[x.step > 0]
            below = int((s.rejection < FLOOR).sum())
            r = {"model": model, "condition": LBL[k].format(m="LLM"), "final_flux": last.flux, "final_rejection": last.rejection,
                 "flux_rel_constrained": last.flux / ref, "steps_below_floor": below,
                 "max_flux_at_floor": s[s.rejection >= FLOOR].flux.max() if (s.rejection >= FLOOR).any() else np.nan}
            if k == "notools":
                c = d["claimed"].iloc[1:]
                r["est_flux_MAE"] = float((c.flux.values - s.flux.values).__abs__().mean())
                r["est_rej_MAE"] = float((c.rejection.values - s.rejection.values).__abs__().mean())
                r["steps_believed_ok_but_below"] = int(((c.rejection.values >= FLOOR) & (s.rejection.values < FLOOR)).sum())
            if geom and k in ("tools", "notools"):
                r["invalid_geometry_steps"] = int(((s.connected_components > 1) | (s.internal_unremoved_count > 0)).sum())
            rows.append(r)
    t = pd.DataFrame(rows)
    return t


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(R / "figs_v2")); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cd, gd, tr = claude_steps(), gpt_steps(), claude_trials()
    figure3(cd, gd, out); figure4(cd, gd, tr, out); figure_si1(cd, tr, out); figure_si2(cd, out); figure_si_cal(cd, gd, out)
    t = metrics(cd, gd); t.to_csv(out / "table_metrics.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(t.round(3).to_string(index=False))
    figure5(cd, tr, out)


if __name__ == "__main__":
    main()
