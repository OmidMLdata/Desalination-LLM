from pathlib import Path

from desalination_llm.geometry import FEATURE_NAMES, build_bond_graph, geometry_features, nearest_candidates
from desalination_llm.pdb import GrapheneStructure


ROOT = Path(__file__).resolve().parents[1]


def test_geometry_features_and_candidates():
    structure = GrapheneStructure.from_pdb(ROOT / "data/structures/pristine_graphene_4040.pdb")
    candidates = nearest_candidates(structure.xy, set(), k=30, edge_margin=2.5)
    assert len(candidates) == 30
    adjacency, pairs = build_bond_graph(structure.xy)
    features = geometry_features(structure.xy, set(candidates[:6]), adjacency, pairs)
    assert list(features) == FEATURE_NAMES
    assert features["removed_atoms_count"] == 6
    assert features["hull_area"] >= 0
    assert features["connected_components"] >= 1
