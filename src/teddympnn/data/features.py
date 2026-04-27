"""Structure parsing and feature computation from PDB/mmCIF files.

Converts structural files into the feature tensor format consumed by
ProteinMPNN and LigandMPNN models.
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
    ATOM_ORDER,
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

# Element symbol → atom type index for LigandMPNN (0-indexed by atomic number)
# 118 elements (H=0 ... Og=117) + unknown (118) = 119 total
_ELEMENTS: tuple[str, ...] = (
    "H",
    "HE",
    "LI",
    "BE",
    "B",
    "C",
    "N",
    "O",
    "F",
    "NE",
    "NA",
    "MG",
    "AL",
    "SI",
    "P",
    "S",
    "CL",
    "AR",
    "K",
    "CA",
    "SC",
    "TI",
    "V",
    "CR",
    "MN",
    "FE",
    "CO",
    "NI",
    "CU",
    "ZN",
    "GA",
    "GE",
    "AS",
    "SE",
    "BR",
    "KR",
    "RB",
    "SR",
    "Y",
    "ZR",
    "NB",
    "MO",
    "TC",
    "RU",
    "RH",
    "PD",
    "AG",
    "CD",
    "IN",
    "SN",
    "SB",
    "TE",
    "I",
    "XE",
    "CS",
    "BA",
    "LA",
    "CE",
    "PR",
    "ND",
    "PM",
    "SM",
    "EU",
    "GD",
    "TB",
    "DY",
    "HO",
    "ER",
    "TM",
    "YB",
    "LU",
    "HF",
    "TA",
    "W",
    "RE",
    "OS",
    "IR",
    "PT",
    "AU",
    "HG",
    "TL",
    "PB",
    "BI",
    "PO",
    "AT",
    "RN",
    "FR",
    "RA",
    "AC",
    "TH",
    "PA",
    "U",
    "NP",
    "PU",
    "AM",
    "CM",
    "BK",
    "CF",
    "ES",
    "FM",
    "MD",
    "NO",
    "LR",
    "RF",
    "DB",
    "SG",
    "BH",
    "HS",
    "MT",
    "DS",
    "RG",
    "CN",
    "NH",
    "FL",
    "MC",
    "LV",
    "TS",
    "OG",
)
NUM_ELEMENT_TYPES: int = 119
UNK_ELEMENT_IDX: int = 118
element_to_idx: dict[str, int] = {sym: i for i, sym in enumerate(_ELEMENTS)}

# Residue names excluded from ligand atom extraction
EXCLUDED_LIGAND_RESIDUES: frozenset[str] = frozenset({"HOH", "WAT", "DOD"})
EXCLUDED_IONS: frozenset[str] = frozenset({"NA", "CL", "K", "BR"})

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


def extract_ligand_atoms(path: str | Path) -> dict[str, torch.Tensor]:
    """Extract non-protein atoms from a PDB or mmCIF file.

    Extracts HETATM records excluding water and common buffer ions,
    returning coordinates, validity masks, and element type indices
    for LigandMPNN context encoding.

    Args:
        path: Path to a structure file.

    Returns:
        Dict with keys:

        - ``Y``: ``(N, 3)`` float32 — non-protein atom coordinates.
        - ``Y_m``: ``(N,)`` bool — atom validity mask.
        - ``Y_t``: ``(N,)`` int64 — element type indices (0–118).
    """
    path = Path(path)
    parser = _get_parser(path)
    structure = parser.get_structure("s", str(path))  # type: ignore[no-untyped-call]
    model = next(structure.get_models())

    coords_list: list[np.ndarray[Any, Any]] = []
    types_list: list[int] = []

    for chain in model:
        for residue in chain:
            het_flag = residue.id[0]
            resname = residue.resname.strip()

            # Skip standard protein residues and modified AAs
            if het_flag == " ":
                continue
            if resname in MODIFIED_AA_MAP:
                continue
            # Skip water and excluded ions
            if resname in EXCLUDED_LIGAND_RESIDUES or resname in EXCLUDED_IONS:
                continue

            for atom in residue:
                elem = atom.element.strip().upper()
                if not elem:
                    continue
                coords_list.append(atom.coord.astype(np.float32))
                types_list.append(element_to_idx.get(elem, UNK_ELEMENT_IDX))

    if not coords_list:
        return {
            "Y": torch.zeros(0, 3, dtype=torch.float32),
            "Y_m": torch.zeros(0, dtype=torch.bool),
            "Y_t": torch.zeros(0, dtype=torch.long),
        }

    coords = np.stack(coords_list)
    return {
        "Y": torch.from_numpy(coords),
        "Y_m": torch.ones(len(coords_list), dtype=torch.bool),
        "Y_t": torch.tensor(types_list, dtype=torch.long),
    }


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


def extract_sidechain_atoms(
    xyz_37: torch.Tensor,
    xyz_37_m: torch.Tensor,
    S: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Extract side-chain atoms from selected residues for LigandMPNN context.

    Gathers resolved side-chain atom coordinates from residues indicated by
    ``mask`` (typically the fixed/conditioning partner), formatted as
    non-protein context atoms (Y/Y_m/Y_t).

    Args:
        xyz_37: All-atom coordinates, shape ``(L, 37, 3)``.
        xyz_37_m: Atom validity mask, shape ``(L, 37)``.
        S: Token indices, shape ``(L,)``.
        mask: Boolean mask of residues to include, shape ``(L,)``.

    Returns:
        Dict with ``Y``, ``Y_m``, ``Y_t`` tensors for side-chain atoms.
    """
    # Side-chain atom indices: 4 (CB) through 35 (NZ), excluding terminal OXT.
    sc_indices = list(range(4, NUM_ATOMS_37 - 1))

    # Gather side-chain coords and masks for selected residues
    selected = mask.bool()
    sc_coords = xyz_37[selected][:, sc_indices, :]  # (M, 32, 3)
    sc_mask = xyz_37_m[selected][:, sc_indices]  # (M, 32)

    # Flatten to (M*32, 3) and filter valid atoms
    flat_coords = sc_coords.reshape(-1, 3)
    flat_mask = sc_mask.reshape(-1)

    valid = flat_mask.bool()
    if not valid.any():
        return {
            "Y": torch.zeros(0, 3, dtype=xyz_37.dtype),
            "Y_m": torch.zeros(0, dtype=torch.bool),
            "Y_t": torch.zeros(0, dtype=torch.long),
        }

    # Map side-chain atom names to element types
    # ATOM_ORDER[4:36] = CB, CG, CG1, ... NZ — extract element from first char
    sc_atom_names = ATOM_ORDER[4:-1]
    sc_element_indices: list[int] = []
    for name in sc_atom_names:
        elem = name[0].upper()  # C, N, O, S from atom names
        sc_element_indices.append(element_to_idx.get(elem, UNK_ELEMENT_IDX))
    elem_per_atom = torch.tensor(sc_element_indices, dtype=torch.long)

    # Tile for all selected residues: (M, 32) → (M*32,)
    num_selected = int(selected.sum().item())
    flat_types = elem_per_atom.unsqueeze(0).expand(num_selected, -1).reshape(-1)

    return {
        "Y": flat_coords[valid],
        "Y_m": torch.ones(int(valid.sum().item()), dtype=torch.bool),
        "Y_t": flat_types[valid],
    }
