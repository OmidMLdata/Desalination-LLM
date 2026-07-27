from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .agents import AgentPanel, NoAgentPanel, votes_to_json
from .config import OptimizationConfig
from .geometry import (
    build_bond_graph,
    colonize_internal_atoms,
    geometry_features,
    nearest_candidates,
)
from .pdb import GrapheneStructure
from .predictors.base import Prediction, Predictor
from .rendering import render_pore


@dataclass(slots=True)
class TrialRecord:
    run: int
    step: int
    trial: int
    removed_atoms_count: int
    flux: float
    rejection: float
    feasible: bool
    accepted: bool
    removed_atom_serials: str
    image_path: str = ""
    pdb_path: str = ""
    agent_votes: str = "[]"


class GraphenePoreOptimizer:
    def __init__(
        self,
        structure: GrapheneStructure,
        predictor: Predictor,
        config: OptimizationConfig,
        agent_panel: AgentPanel | None = None,
    ):
        self.structure = structure
        self.predictor = predictor
        self.config = config
        self.agent_panel = agent_panel or NoAgentPanel()
        self.xy = structure.xy
        self.adjacency, self.bond_pairs = build_bond_graph(self.xy)

    @staticmethod
    def _better(candidate: Prediction, best: Prediction | None, minimum_rejection: float) -> bool:
        if best is None:
            return True
        candidate_feasible = candidate.rejection >= minimum_rejection
        best_feasible = best.rejection >= minimum_rejection
        if candidate_feasible != best_feasible:
            return candidate_feasible
        if candidate_feasible:
            return (candidate.flux, candidate.rejection) > (best.flux, best.rejection)
        return (candidate.rejection, candidate.flux) > (best.rejection, best.flux)

    def run(self, output_dir: str | Path, run_number: int = 0, seed: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(self.config.random_seed if seed is None else seed)
        current_removed: set[int] = set()
        current_prediction = Prediction(0.0, 1.0)
        results: list[dict] = []
        trials: list[TrialRecord] = []

        initial_pdb = self.structure.write_without_indices(output / "step_00.pdb", current_removed)
        initial_image = render_pore(self.xy, current_removed, output / "step_00.png")
        results.append(self._result_row(run_number, 0, current_removed, current_prediction, initial_pdb, initial_image))
        trials.append(self._trial_row(run_number, 0, 0, current_removed, current_prediction, True, True, initial_image, initial_pdb, []))

        for step in range(1, self.config.steps + 1):
            candidate_pool = nearest_candidates(
                self.xy,
                current_removed,
                self.config.nearest_neighbors,
                self.config.edge_margin_angstrom,
            )
            if not candidate_pool or len(current_removed) >= self.config.max_total_removals:
                break

            best_removed: set[int] | None = None
            best_prediction: Prediction | None = None
            best_trial_index: int | None = None
            step_records: list[TrialRecord] = []

            for trial in range(self.config.trials_per_step):
                proposed = set(current_removed)
                available = [idx for idx in candidate_pool if idx not in proposed]
                count = int(rng.integers(0, self.config.max_atoms_per_trial + 1))
                count = min(count, len(available), self.config.max_total_removals - len(proposed))
                if count:
                    proposed.update(rng.choice(available, size=count, replace=False).tolist())
                if self.config.colonize_internal_atoms:
                    proposed = colonize_internal_atoms(
                        self.xy, current_removed, proposed, self.config.max_atoms_per_step
                    )
                if len(proposed - current_removed) > self.config.max_atoms_per_step:
                    new_atoms = sorted(proposed - current_removed)
                    kept = rng.choice(new_atoms, size=self.config.max_atoms_per_step, replace=False).tolist()
                    proposed = current_removed | set(kept)
                prediction = self.predictor.predict(proposed)

                image_path = ""
                pdb_path = ""
                if self.config.save_trial_assets:
                    image = render_pore(self.xy, proposed, output / f"step_{step:02d}_trial_{trial:02d}.png")
                    pdb = self.structure.write_without_indices(output / f"step_{step:02d}_trial_{trial:02d}.pdb", proposed)
                    image_path, pdb_path = image.name, pdb.name
                record = self._trial_row(
                    run_number,
                    step,
                    trial,
                    proposed,
                    prediction,
                    prediction.rejection >= self.config.minimum_rejection,
                    False,
                    image_path,
                    pdb_path,
                    [],
                )
                step_records.append(record)
                if self._better(prediction, best_prediction, self.config.minimum_rejection):
                    best_removed = proposed
                    best_prediction = prediction
                    best_trial_index = trial

            if best_removed is None or best_prediction is None or best_trial_index is None:
                break
            votes = self.agent_panel.review(
                self.structure.removed_serials(best_removed - current_removed),
                best_prediction,
                current_prediction,
            )
            approves = not votes or sum(vote.decision == "REMOVE" for vote in votes) >= 2
            flux_ok = (not self.config.require_flux_improvement) or best_prediction.flux >= current_prediction.flux
            accepted = approves and flux_ok and len(best_removed) > len(current_removed)
            step_records[best_trial_index].accepted = accepted
            step_records[best_trial_index].agent_votes = votes_to_json(votes)
            trials.extend(step_records)

            if accepted:
                current_removed = set(best_removed)
                current_prediction = best_prediction
            step_pdb = self.structure.write_without_indices(output / f"step_{step:02d}.pdb", current_removed)
            step_image = render_pore(self.xy, current_removed, output / f"step_{step:02d}.png", show_hull=True)
            results.append(self._result_row(run_number, step, current_removed, current_prediction, step_pdb, step_image))

        results_frame = pd.DataFrame(results)
        trials_frame = pd.DataFrame([asdict(record) for record in trials])
        results_frame.to_csv(output / "selected_steps.csv", index=False)
        trials_frame.to_csv(output / "all_trials.csv", index=False)
        return results_frame, trials_frame

    def _result_row(
        self,
        run: int,
        step: int,
        removed: set[int],
        prediction: Prediction,
        pdb_path: Path,
        image_path: Path,
    ) -> dict:
        features = geometry_features(
            self.xy, removed, adjacency=self.adjacency, bond_pairs=self.bond_pairs
        )
        return {
            "run": run,
            "step": step,
            "flux": prediction.flux,
            "rejection": prediction.rejection,
            "removed_atoms_count": len(removed),
            "removed_atom_serials": " ".join(map(str, self.structure.removed_serials(removed))),
            "pdb_path": pdb_path.name,
            "image_path": image_path.name,
            **features,
        }

    def _trial_row(
        self,
        run: int,
        step: int,
        trial: int,
        removed: set[int],
        prediction: Prediction,
        feasible: bool,
        accepted: bool,
        image_path: str | Path,
        pdb_path: str | Path,
        votes: list,
    ) -> TrialRecord:
        return TrialRecord(
            run=run,
            step=step,
            trial=trial,
            removed_atoms_count=len(removed),
            flux=prediction.flux,
            rejection=prediction.rejection,
            feasible=feasible,
            accepted=accepted,
            removed_atom_serials=" ".join(map(str, self.structure.removed_serials(removed))),
            image_path=Path(image_path).name if image_path else "",
            pdb_path=Path(pdb_path).name if pdb_path else "",
            agent_votes=votes_to_json(votes),
        )
