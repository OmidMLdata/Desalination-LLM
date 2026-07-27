from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(slots=True)
class AtomRecord:
    serial: int
    name: str
    element: str
    x: float
    y: float
    z: float
    line: str


@dataclass(slots=True)
class GrapheneStructure:
    atoms: list[AtomRecord]
    prefix_lines: list[str]
    suffix_lines: list[str]

    @classmethod
    def from_pdb(cls, path: str | Path) -> "GrapheneStructure":
        atoms: list[AtomRecord] = []
        prefix: list[str] = []
        suffix: list[str] = []
        seen_atom = False
        with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith(("ATOM", "HETATM")):
                    seen_atom = True
                    try:
                        serial = int(line[6:11])
                        name = line[12:16].strip() or "C"
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        element = line[76:78].strip() or name[0]
                    except (ValueError, IndexError):
                        fields = line.split()
                        serial = int(fields[1])
                        name = fields[2]
                        x, y, z = map(float, fields[6:9])
                        element = fields[-1] if fields[-1].isalpha() else name[0]
                    atoms.append(AtomRecord(serial, name, element, x, y, z, line))
                elif not seen_atom:
                    prefix.append(line)
                else:
                    suffix.append(line)
        if not atoms:
            raise ValueError(f"No ATOM/HETATM records found in {path}")
        return cls(atoms=atoms, prefix_lines=prefix, suffix_lines=suffix)

    @property
    def serials(self) -> np.ndarray:
        return np.asarray([atom.serial for atom in self.atoms], dtype=int)

    @property
    def coordinates(self) -> np.ndarray:
        return np.asarray([(atom.x, atom.y, atom.z) for atom in self.atoms], dtype=float)

    @property
    def xy(self) -> np.ndarray:
        return self.coordinates[:, :2]

    def serial_to_index(self) -> dict[int, int]:
        return {atom.serial: idx for idx, atom in enumerate(self.atoms)}

    def write_without_indices(self, output: str | Path, removed_indices: Iterable[int]) -> Path:
        removed = set(int(idx) for idx in removed_indices)
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            handle.writelines(self.prefix_lines)
            for idx, atom in enumerate(self.atoms):
                if idx not in removed:
                    handle.write(atom.line if atom.line.endswith("\n") else atom.line + "\n")
            handle.writelines(self.suffix_lines)
        return output_path

    def removed_indices_in(self, candidate_pdb: str | Path) -> set[int]:
        candidate = GrapheneStructure.from_pdb(candidate_pdb)
        remaining_serials = set(candidate.serials.tolist())
        return {
            idx for idx, serial in enumerate(self.serials.tolist())
            if serial not in remaining_serials
        }

    def removed_serials(self, removed_indices: Iterable[int]) -> list[int]:
        return sorted(self.atoms[int(idx)].serial for idx in removed_indices)
