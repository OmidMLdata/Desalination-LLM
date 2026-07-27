from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def qualifying_designs(
    trials: pd.DataFrame,
    flux_min: float = 150.0,
    flux_max: float = 170.0,
    rejection_min: float = 0.90,
) -> pd.DataFrame:
    return trials.loc[
        trials["flux"].between(flux_min, flux_max, inclusive="both")
        & (trials["rejection"] >= rejection_min)
    ].copy()


def aggregate_runs(selected_steps: pd.DataFrame) -> pd.DataFrame:
    return selected_steps.groupby("step", as_index=False).agg(
        flux_mean=("flux", "mean"),
        flux_std=("flux", "std"),
        rejection_mean=("rejection", "mean"),
        rejection_std=("rejection", "std"),
        removed_atoms_mean=("removed_atoms_count", "mean"),
    )


def reproduce_figures(data_dir: str | Path, output_dir: str | Path) -> list[Path]:
    data_path = Path(data_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trials = pd.read_csv(data_path / "canonical_trials.csv")
    selected = pd.read_csv(data_path / "canonical_selected_steps.csv")
    summary = aggregate_runs(selected)
    created: list[Path] = []

    figure, axis = plt.subplots(figsize=(8, 6))
    for run, group in selected.groupby("run"):
        axis.plot(group["flux"], group["rejection"], marker="o", markersize=3, label=f"Run {run + 1}")
    axis.axvspan(150, 170, alpha=0.12)
    axis.set_xlabel(r"Water flux (ns$^{-1}$)")
    axis.set_ylabel("Ion rejection")
    axis.set_ylim(0, 1.02)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    path = output / "canonical_flux_rejection_trajectories.png"
    figure.savefig(path, dpi=300)
    plt.close(figure)
    created.append(path)

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(summary["step"], summary["flux_mean"], marker="o", label="Flux")
    axis.fill_between(
        summary["step"],
        summary["flux_mean"] - summary["flux_std"].fillna(0),
        summary["flux_mean"] + summary["flux_std"].fillna(0),
        alpha=0.2,
    )
    twin = axis.twinx()
    twin.plot(summary["step"], summary["rejection_mean"], marker="s", label="Rejection")
    twin.fill_between(
        summary["step"],
        summary["rejection_mean"] - summary["rejection_std"].fillna(0),
        summary["rejection_mean"] + summary["rejection_std"].fillna(0),
        alpha=0.15,
    )
    axis.set_xlabel("Design step")
    axis.set_ylabel(r"Water flux (ns$^{-1}$)")
    twin.set_ylabel("Ion rejection")
    twin.set_ylim(0, 1.02)
    figure.tight_layout()
    path = output / "canonical_mean_trajectory.png"
    figure.savefig(path, dpi=300)
    plt.close(figure)
    created.append(path)

    high = trials.loc[trials["rejection"] >= 0.90]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(high["flux"], bins=np.arange(0, 181, 5), alpha=0.85)
    axis.axvspan(150, 170, alpha=0.15)
    axis.set_xlabel(r"Water flux (ns$^{-1}$)")
    axis.set_ylabel("Trial records with rejection ≥ 90%")
    figure.tight_layout()
    path = output / "high_rejection_flux_histogram.png"
    figure.savefig(path, dpi=300)
    plt.close(figure)
    created.append(path)
    return created
