from pathlib import Path

import pandas as pd

from desalination_llm.analysis import qualifying_designs


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_data_window():
    trials = pd.read_csv(ROOT / "data/processed/canonical_trials.csv")
    qualified = qualifying_designs(trials)
    assert len(trials) == 3000
    assert len(qualified) == 28
    assert qualified["removed_atom_serials"].nunique() == 23
    matched = qualified.loc[qualified["design_id"] == "run1_step08_trial17"].iloc[0]
    assert abs(matched["flux"] - 159.88723162098867) < 1e-8
    assert abs(matched["rejection"] - 0.9339835587629484) < 1e-10
