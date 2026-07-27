#!/usr/bin/env python3
"""Validate the compact repository and optionally verify original model assets."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from desalination_llm.analysis import qualifying_designs
from desalination_llm.pdb import GrapheneStructure


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(original_assets: Path | None = None) -> int:
    failures: list[str] = []
    structure = GrapheneStructure.from_pdb(ROOT / "data/structures/pristine_graphene_4040.pdb")
    if len(structure.atoms) != 680:
        failures.append(f"Expected 680 atoms, found {len(structure.atoms)}")
    trials = pd.read_csv(ROOT / "data/processed/canonical_trials.csv")
    qualified = qualifying_designs(trials)
    if len(trials) != 3000:
        failures.append(f"Expected 3000 canonical trials, found {len(trials)}")
    if len(qualified) != 28:
        failures.append(f"Expected 28 qualifying records, found {len(qualified)}")
    if qualified["removed_atom_serials"].nunique() != 23:
        failures.append("Expected 23 unique qualifying removed-atom signatures")
    if not (ROOT / "models/geometry_surrogate.joblib").is_file():
        failures.append("Missing models/geometry_surrogate.joblib")

    if original_assets is not None:
        manifest = pd.read_csv(ROOT / "models/original_asset_manifest.csv")
        for row in manifest.itertuples(index=False):
            filename = Path(row.path).name
            matches = list(original_assets.rglob(filename))
            if not matches:
                failures.append(f"Original asset not found: {filename}")
                continue
            actual = sha256(matches[0])
            if actual != row.sha256:
                failures.append(f"SHA-256 mismatch for {filename}: {actual}")

    if failures:
        print("Repository validation failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("Repository validation passed.")
    print("  pristine atoms: 680")
    print("  canonical trials: 3000")
    print("  qualifying trial records: 28")
    print("  unique qualifying structures: 23")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-assets", type=Path)
    args = parser.parse_args()
    raise SystemExit(validate(args.original_assets))


if __name__ == "__main__":
    main()
