#!/usr/bin/env python3
"""Train the lightweight geometry surrogate from archived CNN/MLP predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import sklearn
import sys
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from desalination_llm.geometry import FEATURE_NAMES


def evaluate(model_factory, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    predictions = np.zeros(len(y), dtype=float)
    for train, test in GroupKFold(n_splits=5).split(X, y, groups):
        model = model_factory()
        model.fit(X[train], y[train])
        predictions[test] = model.predict(X[test])
    return {
        "group_cv_r2": float(r2_score(y, predictions)),
        "group_cv_mae": float(mean_absolute_error(y, predictions)),
        "group_cv_rmse": float(root_mean_squared_error(y, predictions)),
    }


def train(data_path: Path, output_path: Path) -> None:
    frame = pd.read_csv(data_path)
    X = frame[FEATURE_NAMES].to_numpy(dtype=float)
    groups = frame["run"].to_numpy(dtype=int)
    flux_factory = lambda: ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        max_features=0.9,
        n_jobs=-1,
        random_state=42,
    )
    rejection_factory = lambda: HistGradientBoostingRegressor(
        max_iter=300,
        l2_regularization=1.0,
        random_state=42,
    )
    flux_metrics = evaluate(flux_factory, X, frame["flux"].to_numpy(float), groups)
    rejection_metrics = evaluate(rejection_factory, X, frame["rejection"].to_numpy(float), groups)
    flux_model = flux_factory().fit(X, frame["flux"].to_numpy(float))
    rejection_model = rejection_factory().fit(X, frame["rejection"].to_numpy(float))
    metadata = {
        "name": "geometry surrogate of archived CNN/MLP outputs",
        "training_records": int(len(frame)),
        "source": "data/processed/canonical_trials.csv",
        "random_state": 42,
        "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "flux_metrics": flux_metrics,
        "rejection_metrics": rejection_metrics,
        "warning": "Secondary surrogate for runnable demos; not the paper's primary ResNet50+MLP predictor.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "flux_model": flux_model,
        "rejection_model": rejection_model,
        "feature_names": FEATURE_NAMES,
        "metadata": metadata,
    }, output_path, compress=3)
    output_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "canonical_trials.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "geometry_surrogate.joblib")
    args = parser.parse_args()
    train(args.data, args.output)


if __name__ == "__main__":
    main()
