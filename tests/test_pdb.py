from pathlib import Path

from desalination_llm.pdb import GrapheneStructure


ROOT = Path(__file__).resolve().parents[1]


def test_pristine_structure_has_680_atoms(tmp_path):
    structure = GrapheneStructure.from_pdb(ROOT / "data/structures/pristine_graphene_4040.pdb")
    assert len(structure.atoms) == 680
    output = structure.write_without_indices(tmp_path / "removed.pdb", {0, 1, 2})
    candidate = GrapheneStructure.from_pdb(output)
    assert len(candidate.atoms) == 677
    assert structure.removed_indices_in(output) == {0, 1, 2}
