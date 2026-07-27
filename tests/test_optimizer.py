from pathlib import Path

from desalination_llm.agents import RuleBasedPanel
from desalination_llm.config import OptimizationConfig
from desalination_llm.optimizer import GraphenePoreOptimizer
from desalination_llm.pdb import GrapheneStructure
from desalination_llm.predictors.geometry_surrogate import GeometrySurrogatePredictor


ROOT = Path(__file__).resolve().parents[1]


def test_optimizer_smoke(tmp_path):
    structure = GrapheneStructure.from_pdb(ROOT / "data/structures/pristine_graphene_4040.pdb")
    predictor = GeometrySurrogatePredictor(structure.xy, ROOT / "models/geometry_surrogate.joblib")
    config = OptimizationConfig(
        steps=3,
        trials_per_step=4,
        nearest_neighbors=15,
        max_atoms_per_trial=2,
        max_atoms_per_step=2,
        max_total_removals=6,
        minimum_rejection=0.60,
        random_seed=7,
        save_trial_assets=False,
    )
    optimizer = GraphenePoreOptimizer(structure, predictor, config, RuleBasedPanel(0.60))
    results, trials = optimizer.run(tmp_path, seed=7)
    assert len(results) >= 2
    assert len(trials) >= 5
    assert (tmp_path / "selected_steps.csv").is_file()
    assert (tmp_path / "all_trials.csv").is_file()
    assert results["removed_atoms_count"].is_monotonic_increasing
