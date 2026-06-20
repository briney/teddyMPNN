"""Structure parsing and feature computation from PDB/mmCIF files.

Converts structural files into the feature tensor format consumed by
ProteinMPNN models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from Bio.PDB import MMCIFParser, PDBParser  # type: ignore[attr-defined]
from Bio.PDB.Residue import Residue  # noqa: TC002 — used in type annotations

from teddympnn.models.tokens import (
    BACKBONE_ATOM_INDICES,
    NUM_ATOMS_37,
    atom_to_idx,
    token_to_idx,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Standard amino acid 3-letter codes (excludes UNK)
STANDARD_AAS: frozenset[str] = frozenset(token_to_idx.keys()) - {"UNK"}

# Common modified residues → standard parent amino acid
MODIFIED_AA_MAP: dict[str, str] = {
    "MSE": "MET",  # selenomethionine
    "HYP": "PRO",  # hydroxyproline
    "SEP": "SER",  # phosphoserine
    "TPO": "THR",  # phosphothreonine
    "PTR": "TYR",  # phosphotyrosine
    "CSO": "CYS",  # s-hydroxycysteine
    "CSS": "CYS",  # disulfide-linked cysteine
    "CSD": "CYS",  # s-oxy-cysteine
    "MLY": "LYS",  # n-dimethyl-lysine
    "MLZ": "LYS",  # n-monomethyl-lysine
}

# Virtual CB coefficients (identical to graph_embeddings.py)
_CB_A: float = -0.58273431
_CB_B: float = 0.56802827
_CB_C: float = -0.54067466


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_parser(path: Path) -> PDBParser | MMCIFParser:
    """Select the appropriate BioPython parser based on file extension."""
    suffix = path.suffix.lower()
    if suffix in (".pdb", ".ent"):
        return PDBParser(QUIET=True)  # type: ignore[no-untyped-call]
    if suffix in (".cif", ".mmcif"):
        return MMCIFParser(QUIET=True)  # type: ignore[no-untyped-call]
    msg = f"Unsupported structure file format: {suffix}"
    raise ValueError(msg)


def _resolve_resname(resname: str) -> str | None:
    """Map a residue name to a standard amino acid 3-letter code.

    Returns:
        Standard 3-letter code, or ``None`` if not a protein residue.
    """
    resname = resname.strip()
    if resname in STANDARD_AAS:
        return resname
    if resname in MODIFIED_AA_MAP:
        return MODIFIED_AA_MAP[resname]
    return None


def _is_protein_residue(residue: Residue) -> bool:
    """Check if a BioPython residue is a protein residue."""
    het_flag = residue.id[0]
    resname = residue.resname.strip()
    # Standard ATOM records with known AA names
    if het_flag == " ":
        return True
    # Modified residues stored as HETATM
    return resname in MODIFIED_AA_MAP


def _extract_residue_atoms(residue: Residue) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Extract 37-atom coordinates and validity mask from a BioPython residue.

    Returns:
        coords: ``(37, 3)`` float32 array of atom coordinates.
        mask: ``(37,)`` bool array indicating resolved atoms.
    """
    coords = np.zeros((NUM_ATOMS_37, 3), dtype=np.float32)
    mask = np.zeros(NUM_ATOMS_37, dtype=bool)
    for atom in residue:
        name = atom.name.strip()
        if name in atom_to_idx:
            idx = atom_to_idx[name]
            coords[idx] = atom.coord
            mask[idx] = True
    return coords, mask


def _compute_cb(xyz_37: torch.Tensor, xyz_37_m: torch.Tensor) -> torch.Tensor:
    """Compute CB coordinates, using virtual CB where real CB is absent.

    Args:
        xyz_37: All-atom coordinates, shape ``(L, 37, 3)``.
        xyz_37_m: Atom validity mask, shape ``(L, 37)``.

    Returns:
        CB coordinates, shape ``(L, 3)``.
    """
    # Real CB at index 4
    cb_real = xyz_37[:, 4, :]
    cb_present = xyz_37_m[:, 4]

    # Virtual CB from backbone geometry
    n = xyz_37[:, 0, :]
    ca = xyz_37[:, 1, :]
    c = xyz_37[:, 2, :]
    b = ca - n
    cv = c - ca
    a = torch.cross(b, cv, dim=-1)
    cb_virtual = _CB_A * a + _CB_B * b + _CB_C * cv + ca

    return torch.where(cb_present.unsqueeze(-1), cb_real, cb_virtual)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_structure(path: str | Path) -> dict[str, Any]:
    """Parse a PDB or mmCIF file into feature tensors.

    Extracts per-residue coordinates in the 37-atom representation, amino acid
    token indices, chain labels, and residue indices.

    Args:
        path: Path to a ``.pdb``, ``.ent``, ``.cif``, or ``.mmcif`` file.

    Returns:
        Dict with keys:

        - ``xyz_37``: ``(L, 37, 3)`` float32 — all-atom coordinates.
        - ``xyz_37_m``: ``(L, 37)`` bool — atom validity mask.
        - ``S``: ``(L,)`` int64 — amino acid token indices.
        - ``R_idx``: ``(L,)`` int64 — per-chain residue indices (0-based).
        - ``chain_labels``: ``(L,)`` int64 — numeric chain identifiers.
        - ``residue_mask``: ``(L,)`` bool — residue validity mask (all True).
        - ``chain_ids``: ``list[str]`` — original chain ID per residue.
        - ``residue_numbers``: ``list[int]`` — PDB residue sequence numbers.
        - ``residue_icodes``: ``list[str]`` — PDB insertion codes (``""`` if none).
    """
    path = Path(path)
    parser = _get_parser(path)
    structure = parser.get_structure("s", str(path))  # type: ignore[no-untyped-call]
    model = next(structure.get_models())

    all_coords: list[np.ndarray[Any, Any]] = []
    all_masks: list[np.ndarray[Any, Any]] = []
    all_tokens: list[int] = []
    all_chain_labels: list[int] = []
    all_r_idx: list[int] = []
    all_chain_ids: list[str] = []
    all_residue_numbers: list[int] = []
    all_residue_icodes: list[str] = []

    chain_id_map: dict[str, int] = {}

    for chain in model:
        chain_id = chain.id
        if chain_id not in chain_id_map:
            chain_id_map[chain_id] = len(chain_id_map)
        label = chain_id_map[chain_id]

        residue_counter = 0
        for residue in chain:
            if not _is_protein_residue(residue):
                continue

            resname = _resolve_resname(residue.resname)
            if resname is None:
                # Unknown standard ATOM residue — encode as UNK
                resname = "UNK"

            coords, mask = _extract_residue_atoms(residue)

            # Require at least backbone N, CA, C to be valid
            if not (mask[0] and mask[1] and mask[2]):
                logger.debug(
                    "Skipping residue %s:%s — missing backbone N/CA/C",
                    chain_id,
                    residue.id,
                )
                continue

            all_coords.append(coords)
            all_masks.append(mask)
            all_tokens.append(token_to_idx.get(resname, token_to_idx["UNK"]))
            all_chain_labels.append(label)
            all_r_idx.append(residue_counter)
            all_chain_ids.append(chain_id)
            all_residue_numbers.append(residue.id[1])
            icode = residue.id[2]
            all_residue_icodes.append(icode.strip() if isinstance(icode, str) else "")
            residue_counter += 1

    if not all_coords:
        msg = f"No valid protein residues found in {path}"
        raise ValueError(msg)

    return {
        "xyz_37": torch.from_numpy(np.stack(all_coords)),
        "xyz_37_m": torch.from_numpy(np.stack(all_masks)),
        "S": torch.tensor(all_tokens, dtype=torch.long),
        "R_idx": torch.tensor(all_r_idx, dtype=torch.long),
        "chain_labels": torch.tensor(all_chain_labels, dtype=torch.long),
        "residue_mask": torch.ones(len(all_coords), dtype=torch.bool),
        "chain_ids": all_chain_ids,
        "residue_numbers": all_residue_numbers,
        "residue_icodes": all_residue_icodes,
    }


def derive_backbone(
    xyz_37: torch.Tensor,
    xyz_37_m: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract backbone-only coordinates from the 37-atom representation.

    Args:
        xyz_37: All-atom coordinates, shape ``(..., 37, 3)``.
        xyz_37_m: Atom validity mask, shape ``(..., 37)``.

    Returns:
        Tuple of ``(X, X_m)`` where:

        - ``X``: ``(..., 4, 3)`` backbone coordinates [N, CA, C, O].
        - ``X_m``: ``(..., 4)`` backbone atom mask.
    """
    idx = list(BACKBONE_ATOM_INDICES)
    return xyz_37[..., idx, :], xyz_37_m[..., idx]


def identify_interface_residues(
    xyz_37: torch.Tensor,
    xyz_37_m: torch.Tensor,
    chain_labels: torch.Tensor,
    distance_cutoff: float = 8.0,
) -> torch.Tensor:
    """Identify residues at protein-protein interfaces.

    A residue is considered an interface residue if its CB atom (or virtual CB
    for glycine) is within ``distance_cutoff`` of any CB atom on a different
    chain.

    Args:
        xyz_37: All-atom coordinates, shape ``(L, 37, 3)``.
        xyz_37_m: Atom validity mask, shape ``(L, 37)``.
        chain_labels: Chain identifiers, shape ``(L,)``.
        distance_cutoff: CB–CB distance threshold in Angstroms (default 8.0).

    Returns:
        Boolean mask of shape ``(L,)`` — True for interface residues.
    """
    cb = _compute_cb(xyz_37, xyz_37_m)  # (L, 3)

    # Pairwise CB-CB distances: (L, L)
    dist = torch.cdist(cb.unsqueeze(0), cb.unsqueeze(0)).squeeze(0)

    # Cross-chain mask
    cross_chain = chain_labels.unsqueeze(0) != chain_labels.unsqueeze(1)

    # Interface: any cross-chain distance below cutoff
    contacts = (dist < distance_cutoff) & cross_chain
    return contacts.any(dim=1)
