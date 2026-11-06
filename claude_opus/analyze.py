"""Compare the conditions and split the result into model, scaffold and tool contributions.

  python -m claude_opus.analyze

Output: results/claude_opus/{comparison_per_step.csv, summary_metrics.csv,
decomposition.csv, fig_*.png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .env import REPO

R = REPO / "results" / "claude_opus"
MIN_REJ = 0.60

CONDS = {  # tag -> (label, colour, linestyle)
    "opus_tools":           ("Opus 5.5 + tools (agent)",            "#2a78d6", "-"),
    "opus_notools":         ("Opus 5.5 no tools, predictor score",  "#eb6834", "-"),
    "opus_notools_claimed": ("Opus 5.5 no tools, its own estimate", "#eb6834", "--"),
    "scaffold_greedy":      ("Scaffold only, greedy (no LLM)",       "#1baf7a", "-"),
    "scaffold_constrained": ("Scaffold only, greedy + manufacturability rules", "#eda100", "-"),
    "scaffold_random":      ("Scaffold only, random feasible",       "#8a8983", ":"),
}


def load(tag: str) -> pd.DataFrame:
    if tag == "opus_notools_claimed":
        d = pd.read_csv(R / "opus_notools_per_step.csv").copy()
        d.loc[1:, "flux"], d.loc[1:, "rejection"] = d.loc[1:, "claimed_flux"], d.loc[1:, "claimed_rejection"]
        d["condition"] = tag
        return d
    return pd.read_csv(R / f"{tag}_per_step.csv")


def valid(d: pd.DataFrame) -> pd.Series:
    """A design counts as a real pore only if it is one contiguous vacancy cluster with
    no carbon island left inside the hull and it meets the prompt's 60 % rejection limit."""
    return (d.rejection >= MIN_REJ) & (d.connected_components <= 1) & (d.internal_unremoved_count == 0)


def hypervolume(d: pd.DataFrame) -> float:
    """Area of (flux, rejection) space dominated by the valid designs on the trajectory,
    measured from the reference point (flux 0, rejection 0.60). Units: ns^-1 x rejection."""
    pts = d.loc[valid(d) & (d.step > 0), ["flux", "rejection"]].to_numpy()
    if not len(pts):
        return 0.0
    pts = pts[np.argsort(-pts[:, 0])]
    hv, best_r = 0.0, MIN_REJ
    for f, r in pts:
        if r > best_r:
            hv += f * (r - best_r)
            best_r = r
    return hv


def metrics(tag: str, d: pd.DataFrame) -> dict:
    v = valid(d)
    last = d.iloc[-1]
    def best(thr):
        m = v & (d.rejection >= thr) & (d.step > 0)
        return (float(d.loc[m, "flux"].max()), int(d.loc[m, "flux"].idxmax())) if m.any() else (np.nan, -1)
    b90, s90 = best(0.90)
    b60, s60 = best(0.60)
    bad = d.index[(~v) & (d.step > 0)]
    any90 = (d.rejection >= 0.90) & (d.step > 0)   # ignores the manufacturability checks
    return {"condition": tag, "final_flux": last.flux, "final_rejection": last.rejection,
            "final_removed": int(last.removed_count), "final_valid": bool(v.iloc[-1]),
            "best_valid_flux_rej>=0.90": b90, "step_of_best90": d.step.iloc[s90] if s90 >= 0 else np.nan,
            "best_valid_flux_rej>=0.60": b60,
            "best_flux_rej>=0.90_ignoring_geometry": float(d.loc[any90, "flux"].max()) if any90.any() else np.nan,
            "first_invalid_step": int(d.step.iloc[bad[0]]) if len(bad) else np.nan,
            "n_invalid_steps": int(len(bad)), "hypervolume": hypervolume(d)}


def plot_steps(frames, seeds):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, col, lab in [(axes[0], "flux", "Water flux (ns$^{-1}$)"), (axes[1], "rejection", "Ion rejection")]:
        for pol, colr in [("greedy", "#1baf7a"), ("constrained", "#eda100")]:
            if pol in seeds:
                m, s = seeds[pol][col]
                ax.fill_between(m.index, m - s, m + s, color=colr, alpha=0.12, linewidth=0)
        for tag, d in frames.items():
            lab_, c, ls = CONDS[tag]
            ax.plot(d.step, d[col], ls, color=c, lw=2, label=lab_)
            inval = ~valid(d) & (d.step > 0)
            if tag != "opus_notools_claimed" and inval.any():
                ax.scatter(d.step[inval], d[col][inval], s=34, facecolor="white", edgecolor=c, lw=1.6, zorder=3)
        if col == "rejection":
            ax.axhline(MIN_REJ, color="#52514e", lw=1, ls="-.")
            ax.text(0.5, MIN_REJ + 0.01, "60 % limit in the prompt", color="#52514e", fontsize=8)
            ax.axhline(0.90, color="#c3c2b7", lw=0.8)
        ax.set_xlabel("Design step"); ax.set_ylabel(lab)
        ax.grid(alpha=0.25, lw=0.6); ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Per-step water flux and ion rejection (surrogate predictor; hollow markers = invalid design; bands = scaffold seeds 1–5, ±1 sd)", fontsize=10)
    fig.tight_layout(); fig.savefig(R / "fig_per_step.png", dpi=200); plt.close(fig)


def plot_tradeoff(frames):
    fig, ax = plt.subplots(figsize=(6.4, 5))
    for tag, d in frames.items():
        lab, c, ls = CONDS[tag]
        ax.plot(d.flux, d.rejection, ls, color=c, lw=1.6, marker="o", ms=3.5, label=lab)
    ax.axhline(MIN_REJ, color="#52514e", lw=1, ls="-.")
    ax.axvspan(150, 170, color="#c3c2b7", alpha=0.25, lw=0)
    ax.text(151, 0.42, "150–170 ns$^{-1}$\nwindow in the paper", fontsize=8, color="#52514e")
    ax.set_xlabel("Water flux (ns$^{-1}$)"); ax.set_ylabel("Ion rejection")
    ax.grid(alpha=0.25, lw=0.6); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(R / "fig_flux_vs_rejection.png", dpi=200); plt.close(fig)


def plot_calibration(d):
    d = d[d.step > 0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, a, b, lab in [(axes[0], "claimed_flux", "flux", "flux (ns$^{-1}$)"),
                          (axes[1], "claimed_rejection", "rejection", "rejection")]:
        ax.scatter(d[b], d[a], s=26, color="#eb6834", edgecolor="white", lw=0.8)
        lo, hi = min(d[a].min(), d[b].min()), max(d[a].max(), d[b].max())
        ax.plot([lo, hi], [lo, hi], color="#52514e", lw=1)
        for _, r in d.iloc[[0, 9, 19, 29]].iterrows():
            ax.annotate(f"step {int(r.step)}", (r[b], r[a]), fontsize=7, xytext=(4, -10), textcoords="offset points")
        ax.set_xlabel(f"Predictor {lab}"); ax.set_ylabel(f"Opus 5.5's own estimate of {lab}")
        ax.grid(alpha=0.25, lw=0.6); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("No-tool condition: the model's estimates vs the predictor (line = perfect agreement)", fontsize=10)
    fig.tight_layout(); fig.savefig(R / "fig_notools_calibration.png", dpi=200); plt.close(fig)


def plot_pores():
    from scipy.spatial import ConvexHull
    from desalination_llm.pdb import GrapheneStructure
    st = GrapheneStructure.from_pdb(REPO / "data/structures/pristine_graphene_4040.pdb")
    xy, s2i = st.xy, st.serial_to_index()
    tags = ["opus_tools", "opus_notools", "scaffold_greedy", "scaffold_constrained"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.4))
    for ax, t in zip(axes, tags):
        d = load(t); last = d.iloc[-1]
        rem = [s2i[int(v)] for v in str(last.removed_serials).split()] if isinstance(last.removed_serials, str) else []
        c = xy[rem].mean(axis=0) if rem else xy.mean(axis=0)
        keep = np.setdiff1d(np.arange(len(xy)), rem)
        ax.scatter(xy[keep, 0], xy[keep, 1], s=14, c="#0b0b0b", lw=0)
        ax.scatter(xy[rem, 0], xy[rem, 1], s=30, facecolor="none", edgecolor=CONDS[t][1], lw=1.4)
        if len(rem) >= 3:
            h = ConvexHull(xy[rem], qhull_options="QJ"); v = np.r_[h.vertices, h.vertices[0]]
            ax.plot(xy[rem][v, 0], xy[rem][v, 1], color=CONDS[t][1], lw=1.2, alpha=0.7)
        ax.set_xlim(c[0] - 12, c[0] + 12); ax.set_ylim(c[1] - 12, c[1] + 12)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{CONDS[t][0]}\nflux {last.flux:.1f} ns$^{{-1}}$, rejection {last.rejection:.1%}\n"
                     f"{int(last.removed_count)} C removed, {int(last.connected_components)} cluster(s), "
                     f"{int(last.internal_unremoved_count)} island atom(s)", fontsize=8)
    fig.suptitle("Final pores (open circles = removed carbon atoms)", fontsize=10)
    fig.tight_layout(); fig.savefig(R / "fig_final_pores.png", dpi=200); plt.close(fig)


def main():
    frames = {t: load(t) for t in CONDS}
    # seed variability of the scaffold (cheap to rerun; each Opus run is a single seed)
    seeds, seed_rows = {}, []
    for pol in ["greedy", "constrained", "random"]:
        runs = [pd.read_csv(p) for p in sorted(R.glob(f"scaffold_{pol}_s*_per_step.csv"))]
        runs = [load(f"scaffold_{pol}")] + runs
        if len(runs) > 1:
            cat = pd.concat([r.set_index("step")[["flux", "rejection"]] for r in runs], axis=1, keys=range(len(runs)))
            seeds[pol] = {c: (cat.xs(c, axis=1, level=1).mean(axis=1), cat.xs(c, axis=1, level=1).std(axis=1)) for c in ["flux", "rejection"]}
            for i, r in enumerate(runs):
                seed_rows.append({"policy": pol, "run": i, **metrics(f"scaffold_{pol}", r)})

    comp = pd.concat([d.assign(condition=t) for t, d in frames.items()])
    comp[["condition", "step", "atom_serial", "removed_count", "flux", "rejection", "connected_components",
          "internal_unremoved_count", "hull_area", "circularity", "aspect_ratio", "claimed_flux", "claimed_rejection"]] \
        .to_csv(R / "comparison_per_step.csv", index=False)
    summ = pd.DataFrame([metrics(t, d) for t, d in frames.items()])
    summ.to_csv(R / "summary_metrics.csv", index=False)
    if seed_rows:
        sd = pd.DataFrame(seed_rows)
        sd.groupby("policy")[["best_valid_flux_rej>=0.90", "best_valid_flux_rej>=0.60", "hypervolume",
                              "final_flux", "final_rejection", "n_invalid_steps"]].agg(["mean", "std", "count"]) \
          .to_csv(R / "scaffold_seed_summary.csv")

    # ---- calibration of the no-tool self-estimates
    nt = frames["opus_notools"][frames["opus_notools"].step > 0]
    cal = {"flux_MAE": (nt.claimed_flux - nt.flux).abs().mean(),
           "flux_bias(claimed-true)": (nt.claimed_flux - nt.flux).mean(),
           "rejection_MAE": (nt.claimed_rejection - nt.rejection).abs().mean(),
           "rejection_bias(claimed-true)": (nt.claimed_rejection - nt.rejection).mean(),
           "steps_believed_valid_but_rej<0.60": int(((nt.claimed_rejection >= MIN_REJ) & (nt.rejection < MIN_REJ)).sum())}
    pd.Series(cal).to_csv(R / "notools_calibration.csv")

    # ---- decomposition (contrast of two conditions on each metric)
    S = summ.set_index("condition")
    rows = []
    for name, a, b, what in [
        ("Model judgment (tools held fixed)", "opus_tools", "scaffold_greedy", "Opus+tools minus greedy scaffold"),
        ("Model beyond hand-written rules", "opus_tools", "scaffold_constrained", "Opus+tools minus rule-constrained scaffold"),
        ("Tool/predictor feedback (model held fixed)", "opus_tools", "opus_notools", "Opus+tools minus Opus no-tools (predictor-scored)"),
        ("Self-report error without tools", "opus_notools_claimed", "opus_notools", "Opus's own estimates minus predictor"),
    ]:
        for m in ["best_valid_flux_rej>=0.90", "best_valid_flux_rej>=0.60", "hypervolume", "final_flux", "final_rejection", "n_invalid_steps"]:
            rows.append({"effect": name, "contrast": what, "metric": m, "A": S.loc[a, m], "B": S.loc[b, m], "A_minus_B": S.loc[a, m] - S.loc[b, m]})
    pd.DataFrame(rows).to_csv(R / "decomposition.csv", index=False)

    plot_steps({k: v for k, v in frames.items()}, seeds)
    plot_tradeoff({k: v for k, v in frames.items() if k != "scaffold_random"})
    plot_calibration(frames["opus_notools"])
    plot_pores()
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summ.round(3).to_string(index=False)); print(pd.Series(cal).round(3))


if __name__ == "__main__":
    main()
