from pathlib import Path

from desalination_llm.pdb import GrapheneStructure
from desalination_llm.predictors.geometry_surrogate import GeometrySurrogatePredictor


ROOT = Path(__file__).resolve().parents[1]


def test_geometry_surrogate_prediction_ranges():
    structure = GrapheneStructure.from_pdb(ROOT / "data/structures/pristine_graphene_4040.pdb")
    predictor = GeometrySurrogatePredictor(structure.xy, ROOT / "models/geometry_surrogate.joblib")
    baseline = predictor.predict(set())
    trial = predictor.predict({339, 340, 341, 359, 360, 361})
    assert baseline.flux == 0.0
    assert baseline.rejection == 1.0
    assert trial.flux >= 0.0
    assert 0.0 <= trial.rejection <= 1.0
