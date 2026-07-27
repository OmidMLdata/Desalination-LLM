from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Prediction:
    flux: float
    rejection: float


class Predictor(Protocol):
    def predict(self, removed_indices: set[int]) -> Prediction:
        ...
