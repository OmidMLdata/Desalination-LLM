#!/usr/bin/env python3
"""Convert the archived paper run into portable tables and selected structures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from desalination_llm.geometry import FEATURE_NAMES, build_bond_graph, geometry_features
from desalination_llm.pdb import GrapheneStructure


def prepare(canonical_root: Path, output: Path, design_output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    design_output.mkdir(parents=True, exist_ok=True)
    reference = GrapheneStructure.from_pdb(canonical_root / "run_0" / "step_0.pdb")
    serial_to_index = reference.serial_to_index()
    adjacency, pairs = build_bond_graph(reference.xy)
    log = pd.read_csv(canonical_root / "all_runs_trials_log.csv")
    rows: list[dict] = []

    for record in log.itertuples(index=False):
        run = int(record.run)
        pdb_name = Path(record.pdb_path).name
        image_name = Path(record.image_path).name
        candidate = GrapheneStructure.from_pdb(canonical_root / f"run_{run}" / pdb_name)
        remaining = set(candidate.serials.tolist())
        removed_serials = sorted(set(reference.serials.tolist()) - remaining)
        removed_indices = {serial_to_index[serial] for serial in removed_serials}
        features = geometry_features(reference.xy, removed_indices, adjacency, pairs)
        design_id = f"run{run}_step{int(record.step):02d}_trial{int(record.trial):02d}"
        rows.append({
            "design_id": design_id,
            "run": run,
            "step": int(record.step),
            "trial": int(record.trial),
            "flux": float(record.flux),
            "rejection": float(record.rejection),
            "source_pdb": f"run_{run}/{pdb_name}",
            "source_image": f"run_{run}/{image_name}",
            "removed_atom_serials": " ".join(map(str, removed_serials)),
            **features,
        })

    trials = pd.DataFrame(rows)
    trials.to_csv(output / "canonical_trials.csv", index=False)

    selected_frames = []
    for run in range(5):
        path = canonical_root / f"run_{run}" / "intelligent_design_results.csv"
        frame = pd.read_csv(path)
        frame["run"] = run
        frame["pdb_path"] = frame["pdb_path"].map(lambda value: f"run_{run}/{Path(value).name}")
        selected_frames.append(frame)
    selected = pd.concat(selected_frames, ignore_index=True)
    selected.to_csv(output / "canonical_selected_steps.csv", index=False)

    summary = selected.groupby("step", as_index=False).agg(
        flux_mean=("flux", "mean"),
        flux_std=("flux", "std"),
        rejection_mean=("rejection", "mean"),
        rejection_std=("rejection", "std"),
        removed_atoms_count_mean=("removed_atoms_count", "mean"),
    )
    summary.to_csv(output / "canonical_summary.csv", index=False)

    qualifying = trials.loc[
        trials["flux"].between(150.0, 170.0, inclusive="both")
        & (trials["rejection"] >= 0.90)
    ].copy()
    qualifying["structure_signature"] = qualifying["removed_atom_serials"].map(
        lambda value: "-".join(value.split())
    )
    qualifying["is_unique_structure"] = ~qualifying.duplicated("structure_signature")
    asset_pdb = []
    asset_image = []
    for row in qualifying.itertuples(index=False):
        source_pdb = canonical_root / row.source_pdb
        source_image = canonical_root / row.source_image
        destination = design_output / row.design_id
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_pdb, destination / "structure.pdb")
        shutil.copy2(source_image, destination / "structure.png")
        metadata = {
            "design_id": row.design_id,
            "run": int(row.run),
            "step": int(row.step),
            "trial": int(row.trial),
            "flux_ns_inverse": float(row.flux),
            "ion_rejection_fraction": float(row.rejection),
            "removed_atom_serials": [int(value) for value in row.removed_atom_serials.split()],
            "source_pdb": row.source_pdb,
            "source_image": row.source_image,
        }
        (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        asset_pdb.append(f"data/designs/high_performance/{row.design_id}/structure.pdb")
        asset_image.append(f"data/designs/high_performance/{row.design_id}/structure.png")
    qualifying["repository_pdb"] = asset_pdb
    qualifying["repository_image"] = asset_image
    qualifying.to_csv(output / "high_performance_designs.csv", index=False)

    best = qualifying.loc[(qualifying["flux"] - 159.887232).abs().idxmin()]
    best_source = canonical_root / best.source_pdb
    best_image = canonical_root / best.source_image
    final_dir = design_output.parent / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_source, final_dir / "best_reported_design.pdb")
    shutil.copy2(best_image, final_dir / "best_reported_design.png")
    (final_dir / "best_reported_design.json").write_text(json.dumps({
        "design_id": best.design_id,
        "flux_ns_inverse": float(best.flux),
        "ion_rejection_fraction": float(best.rejection),
        "selection_note": "Trial record matching the manuscript-reported ~159.89 ns^-1 and 93.4% rejection design.",
    }, indent=2) + "\n")

    print(f"Wrote {len(trials)} canonical trial records")
    print(f"Wrote {len(qualifying)} qualifying records ({qualifying['structure_signature'].nunique()} unique signatures)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("canonical_root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--design-output", type=Path, default=ROOT / "data" / "designs" / "high_performance")
    args = parser.parse_args()
    prepare(args.canonical_root, args.output, args.design_output)


if __name__ == "__main__":
    main()
