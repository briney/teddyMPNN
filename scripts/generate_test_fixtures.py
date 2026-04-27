"""Generate the small synthetic PDB/mmCIF fixtures used by the test suite.

Produces deterministic two-chain (and ligand-bearing) structures small
enough to commit. Chain identifiers, residue counts, and inter-chain
spacing are chosen so that ``parse_structure`` and
``identify_interface_residues`` produce nontrivial results without
relying on real RCSB downloads.

Run from the repo root::

    python scripts/generate_test_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFIO, PDBIO  # type: ignore[attr-defined]
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure

OUT_DIR = Path(__file__).parent.parent / "tests" / "validation" / "reference_data" / "structures"

# Standard backbone atom offsets (relative to CA at the origin) used to
# fabricate residues that pass the ``parse_structure`` "must have N/CA/C"
# check. Numbers are in Angstroms and approximate canonical bond geometry.
_BACKBONE_OFFSETS: dict[str, tuple[float, float, float]] = {
    "N": (-1.20, 0.55, 0.00),
    "CA": (0.00, 0.00, 0.00),
    "C": (1.20, 0.55, 0.00),
    "O": (1.85, 1.65, 0.00),
    "CB": (-0.30, -1.10, 0.90),
}

# Sequence and 3-letter codes for the synthetic chains. Both chains use a
# repeating residue pattern that includes glycine (no CB) and a couple of
# common amino acids so the parser exercises the standard residue path.
_RESIDUES = ("ALA", "GLY", "VAL", "LEU", "ILE", "SER", "THR", "PHE")


def _make_residue(resname: str, seqid: int, ca_xyz: np.ndarray) -> Residue:
    res = Residue((" ", seqid, " "), resname, "")
    for atom_name, offset in _BACKBONE_OFFSETS.items():
        if atom_name == "CB" and resname == "GLY":
            continue  # Glycine has no CB
        coord = ca_xyz + np.array(offset, dtype=np.float32)
        atom = Atom(
            name=atom_name,
            coord=coord,
            bfactor=20.0,
            occupancy=1.0,
            altloc=" ",
            fullname=f" {atom_name:<3}",
            serial_number=0,  # rewritten by PDBIO
            element=atom_name[0],
        )
        res.add(atom)
    return res


def _make_chain(chain_id: str, *, n_res: int, origin: np.ndarray) -> Chain:
    chain = Chain(chain_id)
    for i in range(n_res):
        # Stretch residues along x at ~3.8 A spacing (canonical Cα-Cα).
        ca = origin + np.array([3.8 * i, 0.0, 0.0], dtype=np.float32)
        resname = _RESIDUES[i % len(_RESIDUES)]
        chain.add(_make_residue(resname, seqid=i + 1, ca_xyz=ca))
    return chain


def _write(structure: Structure, stem: str) -> None:
    pdb_io = PDBIO()
    pdb_io.set_structure(structure)
    pdb_io.save(str(OUT_DIR / f"{stem}.pdb"))

    cif_io = MMCIFIO()
    cif_io.set_structure(structure)
    cif_io.save(str(OUT_DIR / f"{stem}.cif"))


def _build_two_chain() -> Structure:
    """Two interacting chains (A and D) with overlapping x-ranges.

    Chains are placed with a ~5 A separation along y so cross-chain CB-CB
    distances are well within an 8 A interface cutoff.
    """
    structure = Structure("1BRS_mini")
    model = Model(0)
    structure.add(model)
    model.add(_make_chain("A", n_res=8, origin=np.array([0.0, 0.0, 0.0], dtype=np.float32)))
    model.add(_make_chain("D", n_res=7, origin=np.array([0.0, 5.0, 0.0], dtype=np.float32)))
    return structure


def _build_ligand_bearing() -> Structure:
    """One chain plus a small ligand cluster as HETATMs."""
    structure = Structure("4GYT_mini")
    model = Model(0)
    structure.add(model)
    chain = _make_chain("A", n_res=10, origin=np.array([0.0, 0.0, 0.0], dtype=np.float32))

    # Add a 5-atom HETATM "ligand" near the chain. We use a glucose-like
    # element mix (3xC, O, N) to exercise the element parser.
    lig = Residue(("H_LIG", 100, " "), "LIG", "")
    coords = [
        (5.0, 5.0, 0.0, "C"),
        (6.0, 5.5, 0.0, "C"),
        (6.5, 6.5, 0.0, "C"),
        (5.5, 7.0, 0.0, "O"),
        (4.5, 6.0, 0.0, "N"),
    ]
    for i, (x, y, z, elem) in enumerate(coords):
        lig.add(
            Atom(
                name=f"{elem}{i + 1}",
                coord=np.array([x, y, z], dtype=np.float32),
                bfactor=20.0,
                occupancy=1.0,
                altloc=" ",
                fullname=f" {elem}{i + 1:<2}",
                serial_number=0,
                element=elem,
            )
        )
    chain.add(lig)
    model.add(chain)
    return structure


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write(_build_two_chain(), "1BRS_mini")
    _write(_build_ligand_bearing(), "4GYT_mini")
    print(f"Wrote fixtures to {OUT_DIR}")


if __name__ == "__main__":
    main()
