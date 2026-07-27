from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from ..geometry import FEATURE_NAMES, build_bond_graph, geometry_features
from .base import Prediction


class GeometrySurrogatePredictor:
    """Lightweight predictor trained on archived CNN/MLP trial outputs.

    This backend makes the repository runnable without the original neural-network
    checkpoints. It is a secondary surrogate and is not the paper's primary CNN/MLP.
    """

    def __init__(self, xy: np.ndarray, model_path: str | Path):
        payload = joblib.load(model_path)
        self.flux_model = payload["flux_model"]
        self.rejection_model = payload["rejection_model"]
        self.feature_names = payload.get("feature_names", FEATURE_NAMES)
        self.metadata = payload.get("metadata", {})
        self.xy = np.asarray(xy, dtype=float)
        self.adjacency, self.bond_pairs = build_bond_graph(self.xy)

    def predict(self, removed_indices: set[int]) -> Prediction:
        if not removed_indices:
            return Prediction(flux=0.0, rejection=1.0)
        features = geometry_features(
            self.xy,
            removed_indices,
            adjacency=self.adjacency,
            bond_pairs=self.bond_pairs,
        )
        row = np.asarray([[features[name] for name in self.feature_names]], dtype=float)
        flux = float(self.flux_model.predict(row)[0])
        rejection = float(self.rejection_model.predict(row)[0])
        return Prediction(flux=max(0.0, flux), rejection=float(np.clip(rejection, 0.0, 1.0)))
