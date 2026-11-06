"""Manuscript figures, revision 3: Times New Roman 12 pt, the per-step layout of the
original Claude ablation figure for Figure 3, and Figure 4b compared against the dataset
and DRL pores of Wang et al., npj 2D Mater. Appl. 5, 66 (2021).

  python -m claude_opus.figures_v3 --out <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import figures_v2 as V2
from .env import REPO
from .figures_v2 import (C, FLOOR, R, SEEDS, _S2I, _XY, add_thumb, claude_steps, claude_trials,
                         gpt_steps, pore_image, valid)

EXT = REPO / "paper_results" / "reference_wang2021"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
FS = 12
# Times New Roman when installed; Liberation Serif has identical metrics and is the fallback
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "TeX Gyre Termes", "DejaVu Serif"],
    "mathtext.fontset": "custom", "mathtext.rm": "Liberation Serif", "mathtext.it": "Liberation Serif:italic",
    "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FS - 1, "ytick.labelsize": FS - 1,
    "legend.fontsize": FS - 1.5, "figure.titlesize": FS, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "axes.titleweight": "bold"})
MODEL_SHORT = {"Claude Opus 5.5": "Opus 5.5", "GPT-6.1 Sol": "GPT-6.1 Sol"}


def labels(m):
    return {"tools": f"{m} + tools (agent)", "notools": f"{m} no tools, predictor score",
            "claimed": f"{m} no tools, its own estimate", "greedy": "Scaffold only, greedy (no LLM)",
            "constrained": "Scaffold only, greedy + manufacturability rules", "random": "Scaffold only, random feasible"}


def _invalid_steps(df, geometry=True):
    bad = df.rejection < FLOOR
    if geometry and "connected_components" in df:
        bad |= (df.connected_components > 1) | (df.internal_unremoved_count > 0)
    return bad & (df.step > 0)


def _scaffold_invalid(runs, geometry=True):
    """A mean-line step is drawn hollow when most seeds are invalid at that step."""
    m = np.mean([_invalid_steps(r, geometry).values for r in runs], axis=0)
    return pd.Series(m > 0.5)


# ----------------------------------------------------------------------------- Figure 3
def _row(ax_f, ax_r, d, model, geometry, runs_avail, panel):
    L = labels(MODEL_SHORT[model])
    order = [("random", ":", 2.0), ("greedy", "-", 2.2), ("constrained", "-", 2.2)]
    for metric, ax in (("flux", ax_f), ("rejection", ax_r)):
        for p, ls, lw in order:
            x = d[p]
            if f"{metric}_sd" in x:
                ax.fill_between(x.step, x[metric] - x[f"{metric}_sd"], x[metric] + x[f"{metric}_sd"],
                                color=C[p], alpha=0.12, lw=0)
            dash = (0, (6, 2.6)) if (p == "constrained" and model == "GPT-6.1 Sol") else ls
            ax.plot(x.step, x[metric], ls=dash, color=C[p], lw=lw, label=L[p])
            if runs_avail:
                bad = _scaffold_invalid(d[p + "_runs"], geometry).values
                ax.plot(x.step[bad], x[metric][bad], "o", ms=6.5, mfc="white", mec=C[p], mew=1.8, ls="none", zorder=4)
        n, cl, t = d["notools"], d["claimed"], d["tools"]
        ax.plot(n.step, n[metric], color=C["notools"], lw=2.2, label=L["notools"])
        ax.plot(cl.step, cl[metric], color=C["notools"], lw=2.0, ls=(0, (5, 2.4)), label=L["claimed"])
        ax.plot(t.step, t[metric], color=C["tools"], lw=2.4, label=L["tools"])
        for df, c in ((n, C["notools"]), (t, C["tools"])):
            bad = _invalid_steps(df, geometry)
            ax.plot(df.step[bad], df[metric][bad], "o", ms=6.5, mfc="white", mec=c, mew=1.8, ls="none", zorder=5)
        ax.set_xlabel("Design step"); ax.set_xlim(-1, 31); ax.set_xticks(range(0, 31, 5))
    ax_f.set_ylabel("Water flux (ns$^{-1}$)"); ax_f.set_ylim(-5, 245)
    ax_r.set_ylabel("Ion rejection")
    ax_r.axhline(FLOOR, color=INK2, lw=1.1, ls="-.")
    ax_r.text(0.6, FLOOR - 0.016, "90 % rejection floor in the prompt", color=INK2, fontsize=FS - 1.5)
    ax_r.set_ylim(0.6, 1.02)
    h, l = ax_f.get_legend_handles_labels()
    order_l = [L[k] for k in ("tools", "notools", "claimed", "greedy", "constrained", "random")]
    hd = dict(zip(l, h))
    ax_f.legend([hd[k] for k in order_l], order_l, loc="upper left", fontsize=FS - 2, handlelength=2.6)
    ax_f.text(-0.13, 1.02, panel[0], transform=ax_f.transAxes, fontsize=FS + 2, fontweight="bold")
    ax_r.text(-0.13, 1.02, panel[1], transform=ax_r.transAxes, fontsize=FS + 2, fontweight="bold")


def figure3(cd, gd, out):
    fig = plt.figure(figsize=(13.2, 11.2))
    sub = fig.subfigures(2, 1, hspace=0.06)
    for sf, d, model, geom, runs, pl, note in (
            (sub[0], cd, "Claude Opus 5.5", True, True, ("a)", "b)"),
             "CNN-MLP surrogate; hollow markers = invalid design; bands = $\\pm$1 SD over 6 scaffold seeds"),
            (sub[1], gd, "GPT-6.1 Sol", False, False, ("c)", "d)"),
             "CNN-MLP predictor; hollow markers = below the 90 % floor; bands = $\\pm$1 SD over 6 scaffold seeds")):
        a = sf.subplots(1, 2)
        _row(a[0], a[1], d, model, geom, runs, pl)
        sf.suptitle(f"{model}: per-step water flux and ion rejection ({note})", fontsize=FS)
    fig.savefig(out / "Figure3.pdf"); fig.savefig(out / "Figure3.png", dpi=200); plt.close(fig)


# ----------------------------------------------------------------------------- Figure 4
def wang_dataset():
    d = pd.read_csv(EXT / "wang2021_dataset_labels.csv")
    return d[d.rejection > 0.9].flux.values


def wang_drl(total=3428, seed=0):
    """DRL flux distribution (rejection > 90 %) digitised from the published histogram;
    counts rescaled to the reported total and spread uniformly within each bar."""
    b = pd.read_csv(EXT / "wang2021_drl_hist_digitized.csv")
    b = b[b["count"] > 0.5]
    raw = b["count"].values / b["count"].sum() * total
    cnt = np.floor(raw).astype(int)
    cnt[np.argsort(raw - cnt)[::-1][: total - cnt.sum()]] += 1      # largest-remainder rounding
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.uniform(lo, hi, n) for lo, hi, n in zip(b.flux_lo, b.flux_hi, cnt)])


def figure4(cd, gd, tr, out):
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 6.0), gridspec_kw={"width_ratios": [1, 1.08]})
    # (a) performance window: agent run + seed-42 scaffold runs, larger readable markers
    a = ax[0]
    sty = {"random": ("^", 20), "greedy": ("s", 18), "constrained": ("D", 18)}
    for p in ("random", "greedy", "constrained"):
        s = tr[(tr.cond == p) & (tr.seed == 42)]
        a.scatter(s.flux, s.rejection, s=sty[p][1], marker=sty[p][0], color=C[p], alpha=0.45, edgecolor="white", lw=0.3,
                  label=labels("")[p] + " (seed 42)", rasterized=True)
    s = tr[tr.cond == "tools"]
    a.scatter(s.flux, s.rejection, s=26, color=C["tools"], alpha=0.65, edgecolor="white", lw=0.4,
              label="Opus 5.5 + tools, evaluated trials", rasterized=True)
    for k, lab in (("tools", "Opus 5.5 + tools, selected path"), ("notools", "Opus 5.5 no tools, selected path")):
        t = cd[k]
        a.plot(t.flux, t.rejection, "-o", color=C[k], ms=6, lw=2.0, mec="white", mew=0.8, label=lab, zorder=5)
    a.axhline(FLOOR, color=INK2, lw=1.1, ls="-.")
    t = cd["tools"]; v = t[valid(t)]; b = v.loc[v.flux.idxmax()]
    add_thumb(a, (b.flux, b.rejection), pore_image(b.removed_serials), (0.40, 0.42),
              f"Best valid Opus 5.5 design\n{b.flux:.1f} ns$^{{-1}}$, {b.rejection:.1%}", zoom=0.36)
    a.set_xlabel("Water flux (ns$^{-1}$)"); a.set_ylabel("Ion rejection"); a.set_ylim(0.6, 1.0); a.set_xlim(-4, 182)
    a.legend(loc="lower left", fontsize=FS - 2.5, markerscale=1.3)
    a.text(-0.12, 1.03, "a)", transform=a.transAxes, fontsize=FS + 2, fontweight="bold")

    # (b) flux of pores with rejection >= 90 %: dataset and DRL (Wang et al.) vs LLM agents
    a = ax[1]
    ds, drl = wang_dataset(), wang_drl()
    cl = tr[(tr.cond == "tools") & (tr.rejection >= FLOOR) & (tr.n_new > 0)].flux.values
    gt = gd["tools"]; gt = gt[(gt.rejection >= FLOOR) & (gt.step > 0)]
    gp = gt.loc[~gt.flux.round(2).duplicated()].flux.values
    bins = np.arange(0, 182, 7.5)
    series = [(ds, "#f4b183", f"Dataset (MD), Wang et al.: {len(ds)} pores", 0.75),
              (drl, "#9dc3e6", f"DRL agent, Wang et al.: {len(drl)} pores", 0.75),
              (cl, C["tools"], f"Claude Opus 5.5 + tools: {len(cl)} pores", 0.80),
              (gp, "#4a3aa7", f"GPT-6.1 Sol + tools: {len(gp)} designs", 0.80)]
    for vals, col, lab, al in series:
        w = np.full(len(vals), 100.0 / len(vals))
        if col == "#4a3aa7":
            a.hist(vals, bins=bins, weights=w, histtype="step", color=col, lw=2.2, label=lab, zorder=4)
        else:
            a.hist(vals, bins=bins, weights=w, color=col, alpha=al, label=lab, edgecolor="white", lw=0.4)
    a.set_xlabel("Water flux (ns$^{-1}$)"); a.set_ylabel("Pores with rejection $\\geq$ 90 % (% of each set)")
    a.set_xlim(0, 182); a.set_ylim(0, 62)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=FS - 2)
    a.text(-0.10, 1.03, "b)", transform=a.transAxes, fontsize=FS + 2, fontweight="bold")
    # magnified high-flux window placed in empty space above the bars
    lo, hi = 112.5, 172.5  # zoom window
    ins = a.inset_axes([0.38, 0.52, 0.58, 0.44])
    bi = np.arange(lo, hi + 0.1, 7.5)
    for vals, col, lab, al in series:
        sel = vals[(vals >= lo) & (vals < hi)]
        w = np.full(len(sel), 100.0 / len(vals))
        if col == "#4a3aa7":
            ins.hist(sel, bins=bi, weights=w, histtype="step", color=col, lw=2.0, zorder=4)
        else:
            ins.hist(sel, bins=bi, weights=w, color=col, alpha=al, edgecolor="white", lw=0.4)
    ins.set_xlim(lo, hi); ins.tick_params(labelsize=FS - 3); ins.set_title(
        f"High-flux window ({lo:g}–{hi:g} ns$^{{-1}}$)", fontsize=FS - 1.5, fontweight="normal")
    ins.set_facecolor("white"); [sp.set_visible(True) for sp in ins.spines.values()]
    for sp in ins.spines.values():
        sp.set_color(INK2); sp.set_linewidth(0.9)
    a.indicate_inset_zoom(ins, edgecolor=INK2, alpha=0.8)
    fig.tight_layout(w_pad=3)
    fig.savefig(out / "Figure4.pdf", dpi=300); fig.savefig(out / "Figure4.png", dpi=200); plt.close(fig)
    st = pd.DataFrame([{"set": n, "n": len(v), "median": np.median(v), "p90": np.percentile(v, 90), "max": v.max(),
                        "frac_ge_120": (v >= 120).mean()} for n, v in
                       (("dataset", ds), ("drl", drl), ("claude_tools", cl), ("gpt_tools", gp))])
    st.to_csv(out / "fig4b_stats.csv", index=False)
    print(st.round(3).to_string(index=False))


# ----------------------------------------------------------------------------- Figure 5
def thumb(ax, xy, img, box, text, zoom=0.30, fs=FS - 2):
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    ab = AnnotationBbox(OffsetImage(img, zoom=zoom), xy, xybox=box, boxcoords="axes fraction", frameon=True, pad=0.15,
                        bboxprops=dict(edgecolor=INK2, lw=0.7), arrowprops=dict(arrowstyle="-", ls="--", color="#3b1e2e", lw=0.9))
    ax.add_artist(ab)
    below = box[1] >= 0.5
    tb = (box[0], box[1] - 0.125) if below else (box[0], box[1] + 0.125)
    ax.annotate(text, xy=tb, xycoords="axes fraction", ha="center", va="top" if below else "bottom", fontsize=fs,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))


def figure5(cd, tr, out):
    emb = R / "tsne_embedding.csv"
    data = pd.read_csv(emb) if emb.exists() else V2.figure5(cd, tr, out)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.6))
    for a, col, lab, fmt, panel in [(ax[0], "flux", "Water flux (ns$^{-1}$)", "{:.1f} ns$^{{-1}}$", "a)"),
                                    (ax[1], "rejection", "Ion rejection", "{:.1%}", "b)")]:
        sc = a.scatter(data.t1, data.t2, c=data[col], cmap="cividis", s=14, lw=0, alpha=0.9, rasterized=True)
        cb = fig.colorbar(sc, ax=a, fraction=0.045, pad=0.02); cb.set_label(lab, fontsize=FS)
        cb.ax.tick_params(labelsize=FS - 1)
        if col == "rejection":
            cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        for cond in ("tools", "notools"):
            sel = cd[cond].iloc[1:]
            pts = data[(data.cond == cond) & data.removed_serials.isin(sel.removed_serials)]
            pts = pts.drop_duplicates("removed_serials").set_index("removed_serials").reindex(sel.removed_serials.drop_duplicates()).dropna(subset=["t1"])
            a.plot(pts.t1, pts.t2, "-o", color=C[cond], ms=7, lw=1.8, mec="white", mew=0.9, zorder=4)
        vs = data[(data.cond == "constrained") & valid(data)]
        picks = [data[data.f_removed_atoms_count > 0].sort_values("f_removed_atoms_count").iloc[0],
                 data[(data.cond == "tools") & valid(data)].sort_values("flux").iloc[-1],
                 data[data.cond == "notools"].sort_values("step").iloc[-1],
                 data.loc[data.flux.idxmax()], vs.loc[vs.flux.idxmax()],
                 data.loc[(data.flux - data.flux.median()).abs().idxmin()]]
        slots = [(0.10, 0.85), (0.37, 0.87), (0.63, 0.87), (0.90, 0.85), (0.10, 0.13), (0.37, 0.11), (0.63, 0.11), (0.90, 0.13)]
        xr = (data.t1.min(), data.t1.max()); yr = (data.t2.min(), data.t2.max()); used = set()
        for r in picks:
            fx = (r.t1 - xr[0]) / (xr[1] - xr[0]); fy = (r.t2 - yr[0]) / (yr[1] - yr[0])
            j = min((i for i in range(len(slots)) if i not in used), key=lambda i: (slots[i][0] - fx) ** 2 + (slots[i][1] - fy) ** 2)
            used.add(j); thumb(a, (r.t1, r.t2), pore_image(r.removed_serials), slots[j], fmt.format(r[col]), zoom=0.30)
        a.set_xlabel("t-SNE dimension 1"); a.set_ylabel("t-SNE dimension 2"); a.grid(False); a.set_xticks([]); a.set_yticks([])
        for sp in ("top", "right"):
            a.spines[sp].set_visible(True)
        a.set_xlim(xr[0] - 0.15 * (xr[1] - xr[0]), xr[1] + 0.15 * (xr[1] - xr[0]))
        a.set_ylim(yr[0] - 0.33 * (yr[1] - yr[0]), yr[1] + 0.33 * (yr[1] - yr[0]))
        a.text(-0.06, 1.02, panel, transform=a.transAxes, fontsize=FS + 2, fontweight="bold")
    h = [Line2D([], [], color=C["tools"], marker="o", ms=7, lw=1.8, label="Claude Opus 5.5 + tools, selected pores"),
         Line2D([], [], color=C["notools"], marker="o", ms=7, lw=1.8, label="Claude Opus 5.5 no tools, selected pores")]
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=FS - 1, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(w_pad=2)
    fig.savefig(out / "Figure5.pdf", dpi=300); fig.savefig(out / "Figure5.png", dpi=200); plt.close(fig)


# ----------------------------------------------------------------------------- SI figures
def figure_si1(cd, tr, out):
    fig = plt.figure(figsize=(14, 8.4))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.25, 1], hspace=0.38)
    a = fig.add_subplot(gs[0, :])
    sty = {"random": ("^", 16), "greedy": ("s", 14), "constrained": ("D", 14)}
    for p in ("random", "greedy", "constrained"):
        x = tr[tr.cond == p]
        a.scatter(x.flux, x.rejection, s=sty[p][1], marker=sty[p][0], color=C[p], alpha=0.30, edgecolor="none", rasterized=True)
    x = tr[tr.cond == "tools"]
    a.scatter(x.flux, x.rejection, s=22, color=C["tools"], alpha=0.7, edgecolor="white", lw=0.3, rasterized=True)
    a.axhline(FLOOR, color=INK2, lw=1.1, ls="-.")
    a.set_xlabel("Water flux (ns$^{-1}$)"); a.set_ylabel("Ion rejection"); a.set_ylim(0.45, 1.01)
    L = labels("")
    a.legend(handles=[Line2D([], [], marker="o", ls="", ms=7, color=C["tools"], label="Claude Opus 5.5 + tools (900 trials)")] +
             [Line2D([], [], marker=sty[p][0], ls="", ms=7, color=C[p], label=L[p] + " (6 seeds, 5400 trials)") for p in ("greedy", "constrained", "random")],
             loc="lower left", fontsize=FS - 2)
    a.text(-0.06, 1.03, "a)", transform=a.transAxes, fontsize=FS + 2, fontweight="bold")
    final = [("Opus 5.5 + tools", cd["tools"].iloc[-1], C["tools"]), ("Opus 5.5 no tools", cd["notools"].iloc[-1], C["notools"])]
    for p, nm in (("greedy", "Greedy"), ("constrained", "Greedy + rules"), ("random", "Random")):
        final.append((f"Scaffold: {nm} (seed 42)", pd.read_csv(R / f"scaffold_{p}_s42_per_step.csv").iloc[-1], C[p]))
    for i, (name, r, c) in enumerate(final):
        b = fig.add_subplot(gs[1, i])
        rem = [_S2I[int(s)] for s in str(r.removed_serials).split()] if isinstance(r.removed_serials, str) else []
        cen = _XY[rem].mean(0) if rem else _XY.mean(0)
        keep = np.setdiff1d(np.arange(len(_XY)), rem)
        b.scatter(_XY[keep, 0], _XY[keep, 1], s=16, c="#5ec4c4", edgecolor="#2b8c8c", lw=0.3)
        b.scatter(_XY[rem, 0], _XY[rem, 1], s=24, facecolor="none", edgecolor=c, lw=1.4)
        b.set_xlim(cen[0] - 10, cen[0] + 10); b.set_ylim(cen[1] - 10, cen[1] + 10); b.set_aspect("equal"); b.axis("off")
        ok = "valid" if (r.rejection >= FLOOR and r.connected_components <= 1 and r.internal_unremoved_count == 0) else "invalid"
        b.set_title(f"{name}\n{r.flux:.1f} ns$^{{-1}}$, {r.rejection:.1%}\n{int(r.removed_count)} C, "
                    f"{int(r.connected_components)} cluster(s), {ok}", fontsize=FS - 2, fontweight="normal")
    fig.text(0.07, 0.43, "b)", fontsize=FS + 2, fontweight="bold")
    fig.savefig(out / "FigureSI1.pdf", dpi=300); fig.savefig(out / "FigureSI1.png", dpi=200); plt.close(fig)


def figure_si2(cd, gd, out):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    L = labels("")
    for row, (d, model) in enumerate(((cd, "Claude Opus 5.5"), (gd, "GPT-6.1 Sol"))):
        for p, dx in (("greedy", -0.2), ("constrained", 0.0), ("random", 0.2)):
            x = d[p]
            if row == 1 and p == "constrained":
                dx = 0.0
            for col, m in enumerate(("flux", "rejection")):
                ax[row, col].errorbar(x.step + dx, x[m], yerr=x.get(f"{m}_sd"), fmt="o", ms=4.5, color=C[p], ecolor=C[p],
                                      elinewidth=1.0, capsize=2, label=L[p], mfc="white" if (row == 1 and p == "constrained") else C[p])
        ax[row, 0].set_ylabel("Water flux (ns$^{-1}$)"); ax[row, 1].set_ylabel("Ion rejection")
        ax[row, 0].set_ylim(-5, 235)
        ax[row, 1].axhline(FLOOR, color=INK2, lw=1.1, ls="-.")
        for col in range(2):
            ax[row, col].set_xlabel("Design step")
            ax[row, col].text(-0.12, 1.04, "abcd"[2 * row + col] + ")", transform=ax[row, col].transAxes, fontsize=FS + 2, fontweight="bold")
        ax[row, 0].set_title(f"{model} protocol: water flux, mean $\\pm$ SD (6 seeds)", fontsize=FS)
        ax[row, 1].set_title(f"{model} protocol: ion rejection, mean $\\pm$ SD (6 seeds)", fontsize=FS)
        ax[row, 0].legend(loc="upper left", fontsize=FS - 2)
    fig.tight_layout(h_pad=2.5)
    fig.savefig(out / "Figure2SI.pdf"); fig.savefig(out / "Figure2SI.png", dpi=200); plt.close(fig)


def figure_si_cal(cd, gd, out):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    for d, m, mk, col in ((cd, "Claude Opus 5.5", "o", C["notools"]), (gd, "GPT-6.1 Sol", "s", "#4a3aa7")):
        p, c = d["notools"].iloc[1:], d["claimed"].iloc[1:]
        for a, k in zip(ax, ("flux", "rejection")):
            a.scatter(p[k], c[k], s=60, marker=mk, color=col, edgecolor="white", lw=0.8, label=m, zorder=3)
    for a, k, lab, panel in zip(ax, ("flux", "rejection"), ("water flux (ns$^{-1}$)", "ion rejection"), ("a)", "b)")):
        lim = [min(a.get_xlim()[0], a.get_ylim()[0]), max(a.get_xlim()[1], a.get_ylim()[1])]
        a.plot(lim, lim, color=INK2, lw=1.1, label="Perfect agreement"); a.set_xlim(lim); a.set_ylim(lim)
        a.set_xlabel(f"Predictor {lab}"); a.set_ylabel(f"Model's own estimate of {lab}")
        a.text(-0.12, 1.03, panel, transform=a.transAxes, fontsize=FS + 2, fontweight="bold")
    ax[1].axhline(FLOOR, color=INK2, ls="-.", lw=1.0); ax[1].axvline(FLOOR, color=INK2, ls="-.", lw=1.0)
    ax[0].legend(loc="upper left")
    fig.tight_layout(w_pad=3)
    fig.savefig(out / "FigureSI_calibration.pdf"); fig.savefig(out / "FigureSI_calibration.png", dpi=200); plt.close(fig)


def table_si_steps(cd, gd, out):
    rows = []
    for s in range(31):
        r = {"step": s}
        for tag, d in (("claude", cd), ("gpt", gd)):
            for k in ("tools", "notools", "claimed"):
                x = d[k].set_index("step")
                r[f"{tag}_{k}_flux"], r[f"{tag}_{k}_rej"] = x.loc[s, "flux"], x.loc[s, "rejection"]
        rows.append(r)
    t = pd.DataFrame(rows); t.to_csv(out / "tableS2_per_step.csv", index=False)
    return t


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(R / "figs_v3"))
    ap.add_argument("--skip-tsne", action="store_true"); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cd, gd, tr = claude_steps(), gpt_steps(), claude_trials()
    figure3(cd, gd, out); figure4(cd, gd, tr, out); figure_si1(cd, tr, out); figure_si2(cd, gd, out)
    figure_si_cal(cd, gd, out); table_si_steps(cd, gd, out)
    figure5(cd, tr, out)


if __name__ == "__main__":
    main()
