"""Graphene nanopore design environment shared by every experimental condition.

Every condition (scaffold-only, Claude + tools, Claude conversation-only) runs through
this one class, so they all get the same starting sheet, the same candidate
neighbourhood (k nearest unremoved atoms, sheet edges excluded), the same seeded trial
generator and the same predictor. The only things that change between conditions are
*who chooses the next pore* and *what that decision-maker is allowed to see*.

Two protocols are provided:
  fig3_v1 : 20 trials/step, 1 atom/trial, rejection floor 0.60 (first Claude ablation)
  gpt61   : 30 trials/step, 0-3 atoms/trial + colonisation (<=3 new atoms/step),
            rejection floor 0.90 (matches the GPT-6.1 run results_gpt61_t30_g3_20261015)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from desalination_llm.geometry import (build_bond_graph, colonize_internal_atoms,
                                        geometry_features, nearest_candidates)
from desalination_llm.pdb import GrapheneStructure
from desalination_llm.predictors.base import Prediction
from desalination_llm.rendering import render_pore

REPO = Path(__file__).resolve().parents[1]
DEFAULT_PDB = REPO / "data/structures/pristine_graphene_4040.pdb"

PROTOCOLS = {
    "fig3_v1": dict(steps=30, trials_per_step=20, k_neighbors=30, max_atoms_per_trial=1,
                    max_new_atoms_per_step=1, min_rejection=0.60, colonize=False, allow_zero=False),
    "gpt61":   dict(steps=30, trials_per_step=30, k_neighbors=30, max_atoms_per_trial=3,
                    max_new_atoms_per_step=3, min_rejection=0.90, colonize=True, allow_zero=True),
}


@dataclass
class DesignConfig:
    protocol: str = "gpt61"
    steps: int = 30
    trials_per_step: int = 30
    k_neighbors: int = 30
    max_atoms_per_trial: int = 3
    max_new_atoms_per_step: int = 3
    min_rejection: float = 0.90
    colonize: bool = True
    allow_zero: bool = True          # a trial may remove 0 atoms (= keep the current pore)
    edge_margin: float = 2.5
    seed: int = 42
    backend: str = "geometry-surrogate"   # or "cnn-mlp"
    model_path: str = str(REPO / "models/geometry_surrogate.joblib")
    cnn_model_dir: str = str(REPO / "models/original_cnn_mlp")
    cnn_labels_csv: str = str(REPO / "models/original_cnn_mlp/noisy_train.csv")

    @classmethod
    def preset(cls, protocol: str = "gpt61", **kw) -> "DesignConfig":
        return cls(protocol=protocol, **{**PROTOCOLS[protocol], **kw})


def make_predictor(cfg: DesignConfig, xy: np.ndarray):
    if cfg.backend == "cnn-mlp":
        from desalination_llm.predictors.cnn_mlp import CNNMLPPredictor
        return CNNMLPPredictor(xy, cfg.cnn_model_dir, cfg.cnn_labels_csv)
    from desalination_llm.predictors.geometry_surrogate import GeometrySurrogatePredictor
    return GeometrySurrogatePredictor(xy, cfg.model_path)


SHAPE_KEYS = ("hull_area", "hull_perimeter", "circularity", "aspect_ratio",
              "connected_components", "boundary_atoms", "internal_unremoved_count")


@dataclass
class DesalEnv:
    cfg: DesignConfig = field(default_factory=DesignConfig.preset)
    pdb_path: str = str(DEFAULT_PDB)
    blind: bool = False   # True => predictor outputs are never shown to the decision-maker

    def __post_init__(self):
        self.structure = GrapheneStructure.from_pdb(self.pdb_path)
        self.xy = self.structure.xy
        self.serials = self.structure.serials
        self.s2i = self.structure.serial_to_index()
        self.adjacency, self.bond_pairs = build_bond_graph(self.xy)
        self.predictor = make_predictor(self.cfg, self.xy)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.removed: set[int] = set()
        self.step = 0
        self.current = Prediction(0.0, 1.0)
        self.history: list[dict] = [self._row([], self.current, note="initial pristine sheet")]
        self.trial_log: list[dict] = []          # every evaluated trial (for t-SNE / histograms)
        self._trials: list[dict] | None = None
        self._trial_sets: list[set[int]] | None = None
        self.n_predictor_calls = 0

    def __getstate__(self):          # pickle without the (large) predictor; reloaded on unpickle
        d = dict(self.__dict__); d.pop("predictor", None); return d

    def __setstate__(self, d):
        self.__dict__.update(d); self.predictor = make_predictor(self.cfg, self.xy)

    # ------------------------------------------------------------------ helpers
    def _predict(self, removed: set[int]) -> Prediction:
        self.n_predictor_calls += 1
        if not removed:
            return Prediction(0.0, 1.0)
        return self.predictor.predict(set(removed))

    def _features(self, removed: set[int]) -> dict:
        return geometry_features(self.xy, removed, self.adjacency, self.bond_pairs)

    def _shape(self, removed: set[int]) -> dict:
        f = self._features(removed)
        return {k: round(float(f[k]), 3) for k in SHAPE_KEYS}

    def _ser(self, idxs) -> list[int]:
        return sorted(int(self.serials[i]) for i in idxs)

    def _row(self, new_serials, pred: Prediction, **extra) -> dict:
        return {"step": self.step, "new_atoms": " ".join(map(str, new_serials)), "removed_count": len(self.removed),
                "flux": pred.flux, "rejection": pred.rejection,
                "removed_serials": " ".join(map(str, self._ser(self.removed))),
                **self._shape(self.removed), **extra}

    def done(self) -> bool:
        return self.step >= self.cfg.steps

    # ------------------------------------------------------------------ observations
    def candidate_pool(self) -> list[int]:
        return nearest_candidates(self.xy, self.removed, self.cfg.k_neighbors, self.cfg.edge_margin)

    def describe_candidates(self) -> list[dict]:
        """Geometry-only view of the k-neighbour pool (no predictor call)."""
        return [{"serial": int(self.serials[i]), "x": round(float(self.xy[i, 0]), 3), "y": round(float(self.xy[i, 1]), 3),
                 "bonds_to_pore": sum(1 for j in self.adjacency[i] if j in self.removed)}
                for i in self.candidate_pool()]

    def state(self) -> dict:
        s = {"step": self.step, "max_steps": self.cfg.steps, "removed_count": len(self.removed),
             "removed_serials": self._ser(self.removed), "shape": self._shape(self.removed)}
        if not self.blind:
            s.update(current_flux=round(self.current.flux, 3), current_rejection=round(self.current.rejection, 4))
        return s

    # ------------------------------------------------------------------ trial generation (the scaffold)
    def _propose(self) -> set[int]:
        c = self.cfg
        pool = [i for i in self.candidate_pool() if i not in self.removed]
        lo = 0 if c.allow_zero else 1
        n = min(int(self.rng.integers(lo, c.max_atoms_per_trial + 1)), len(pool))
        proposed = set(self.removed)
        if n:
            proposed.update(self.rng.choice(pool, size=n, replace=False).tolist())
        if c.colonize:
            proposed = colonize_internal_atoms(self.xy, self.removed, proposed, c.max_new_atoms_per_step)
        new = sorted(proposed - self.removed)
        if len(new) > c.max_new_atoms_per_step:
            proposed = self.removed | set(self.rng.choice(new, size=c.max_new_atoms_per_step, replace=False).tolist())
        return proposed

    def evaluate_trials(self) -> list[dict]:
        if self.blind:
            raise PermissionError("Predictor access is disabled in the conversation-only condition.")
        if self._trials is None:
            sets, out = [], []
            for t in range(1, self.cfg.trials_per_step + 1):
                prop = self._propose()
                p = self._predict(prop)
                new = self._ser(prop - self.removed)
                rec = {"trial": t, "new_atoms": new, "n_new": len(new), "flux": round(p.flux, 3),
                       "rejection": round(p.rejection, 4), "feasible": p.rejection >= self.cfg.min_rejection,
                       **self._shape(prop)}
                sets.append(prop); out.append(rec)
                self.trial_log.append({"step": self.step + 1, **{k: v for k, v in rec.items() if k != "new_atoms"},
                                       "new_atoms": " ".join(map(str, new)),
                                       "removed_serials": " ".join(map(str, self._ser(prop))),
                                       **{f"f_{k}": float(v) for k, v in self._features(prop).items()}})
            self._trials, self._trial_sets = out, sets
        return self._trials

    # ------------------------------------------------------------------ actions
    def _apply(self, proposed: set[int], extra: dict) -> dict:
        new = self._ser(proposed - self.removed)
        pred = self._predict(proposed)
        self.step += 1
        self.removed, self.current = set(proposed), pred
        self._trials = self._trial_sets = None
        self.history.append(self._row(new, pred, action="remove" if new else "keep", **extra))
        return {"accepted": bool(new), "new_atoms": new, **self.state()}

    def keep(self, **extra) -> dict:
        self.step += 1
        self._trials = self._trial_sets = None
        self.history.append(self._row([], self.current, action="keep", **extra))
        return {"accepted": False, **self.state()}

    def commit_trial(self, trial: int, *, votes=None, rationale: str = "") -> dict:
        """Tool condition: accept trial number `trial` (1-based) of this step, or -1 to keep."""
        votes = votes or []
        extra = {"votes": json.dumps(votes), "rationale": rationale}
        n_remove = sum(1 for v in votes if str(v.get("decision", "")).upper() == "REMOVE")
        if trial is None or int(trial) < 0 or (votes and n_remove < 2):
            return self.keep(**extra)
        trials = self.evaluate_trials()
        if not 1 <= int(trial) <= len(trials):
            raise ValueError(f"trial must be 1..{len(trials)}")
        t = trials[int(trial) - 1]
        if t["rejection"] < self.cfg.min_rejection:
            raise ValueError(f"trial {trial} rejection {t['rejection']:.3f} < floor {self.cfg.min_rejection}")
        return self._apply(self._trial_sets[int(trial) - 1], extra)

    def commit_atoms(self, serials: list[int], *, votes=None, rationale: str = "",
                     claimed_flux=None, claimed_rejection=None) -> dict:
        """Conversation condition: remove up to max_new_atoms_per_step atoms chosen from the
        k-neighbour pool. The predictor scores the result but the score is not returned."""
        votes = votes or []
        extra = {"votes": json.dumps(votes), "rationale": rationale,
                 "claimed_flux": claimed_flux, "claimed_rejection": claimed_rejection}
        serials = [int(s) for s in serials if int(s) >= 0]
        n_remove = sum(1 for v in votes if str(v.get("decision", "")).upper() == "REMOVE")
        if not serials or (votes and n_remove < 2):
            return self.keep(**extra)
        if len(serials) > self.cfg.max_new_atoms_per_step:
            raise ValueError(f"at most {self.cfg.max_new_atoms_per_step} atoms per step")
        pool = set(self.candidate_pool())
        idx = []
        for s in serials:
            i = self.s2i.get(s)
            if i is None or i in self.removed:
                raise ValueError(f"atom {s} does not exist or is already removed")
            if i not in pool:
                raise ValueError(f"atom {s} is not in the k={self.cfg.k_neighbors} neighbourhood")
            idx.append(i)
        res = self._apply(self.removed | set(idx), extra)
        return {k: v for k, v in res.items()} if not self.blind else {k: v for k, v in res.items()
                                                                       if k not in ("current_flux", "current_rejection")}

    # ------------------------------------------------------------------ persistence
    def save(self, out_dir: str | Path, tag: str) -> Path:
        import pandas as pd
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.history); df.insert(0, "condition", tag)
        df.to_csv(out / f"{tag}_per_step.csv", index=False)
        if self.trial_log:
            pd.DataFrame(self.trial_log).assign(condition=tag).to_csv(out / f"{tag}_trials.csv", index=False)
        self.structure.write_without_indices(out / f"{tag}_final.pdb", self.removed)
        render_pore(self.xy, self.removed, out / f"{tag}_final.png", show_hull=True)
        return out / f"{tag}_per_step.csv"
