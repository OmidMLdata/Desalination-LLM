from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class OptimizationConfig:
    steps: int = 30
    trials_per_step: int = 20
    nearest_neighbors: int = 30
    max_atoms_per_trial: int = 3
    max_atoms_per_step: int = 3
    max_total_removals: int = 90
    minimum_rejection: float = 0.60
    random_seed: int = 42
    edge_margin_angstrom: float = 2.5
    colonize_internal_atoms: bool = True
    require_flux_improvement: bool = False
    save_trial_assets: bool = False

    def validate(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        if self.trials_per_step < 1:
            raise ValueError("trials_per_step must be >= 1")
        if self.nearest_neighbors < 1:
            raise ValueError("nearest_neighbors must be >= 1")
        if self.max_atoms_per_trial < 0:
            raise ValueError("max_atoms_per_trial must be >= 0")
        if self.max_atoms_per_step < 0:
            raise ValueError("max_atoms_per_step must be >= 0")
        if not 0.0 <= self.minimum_rejection <= 1.0:
            raise ValueError("minimum_rejection must be between 0 and 1")


@dataclass(slots=True)
class PredictorConfig:
    backend: str = "geometry-surrogate"
    model_path: str = "models/geometry_surrogate.joblib"
    model_dir: str = "models/original_cnn_mlp"
    labels_csv: str = "data/external/noisy_train.csv"
    device: str = "auto"


@dataclass(slots=True)
class AgentConfig:
    mode: str = "rule-based"
    provider_model: str = "gpt-4o-mini"
    minimum_votes: int = 2
    command: str = "gemini -p"


@dataclass(slots=True)
class ProjectConfig:
    structure: str = "data/structures/pristine_graphene_4040.pdb"
    output_dir: str = "results/demo"
    runs: int = 1
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)

    def validate(self) -> None:
        self.optimization.validate()
        if self.runs < 1:
            raise ValueError("runs must be >= 1")


def _merge_dataclass(cls: type, values: dict[str, Any] | None):
    values = values or {}
    allowed = {field.name for field in cls.__dataclass_fields__.values()}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} field(s): {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    defaults = ProjectConfig()
    config = ProjectConfig(
        structure=raw.get("structure", defaults.structure),
        output_dir=raw.get("output_dir", defaults.output_dir),
        runs=int(raw.get("runs", 1)),
        optimization=_merge_dataclass(OptimizationConfig, raw.get("optimization")),
        predictor=_merge_dataclass(PredictorConfig, raw.get("predictor")),
        agents=_merge_dataclass(AgentConfig, raw.get("agents")),
    )
    config.validate()
    return config
