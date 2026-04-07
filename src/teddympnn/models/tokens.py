"""Amino acid vocabulary, atom ordering, and legacy/current format conversions.

Foundry's "current" format uses 3-letter alphabetical token ordering and
outer-product atom pair ordering. The "legacy" format (dauparas/ProteinMPNN)
uses 1-letter alphabetical token ordering and same-atom-first pair ordering.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Amino acid vocabulary
# ---------------------------------------------------------------------------

AMINO_ACIDS_3TO1: dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "UNK": "X",
}

AMINO_ACIDS_1TO3: dict[str, str] = {v: k for k, v in AMINO_ACIDS_3TO1.items()}

# Current Foundry format: 3-letter alphabetical
TOKEN_ORDER: tuple[str, ...] = (
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
)

# Legacy dauparas format: 1-letter alphabetical
LEGACY_TOKEN_ORDER: tuple[str, ...] = (
    "ALA",  # A
    "CYS",  # C
    "ASP",  # D
    "GLU",  # E
    "PHE",  # F
    "GLY",  # G
    "HIS",  # H
    "ILE",  # I
    "LYS",  # K
    "LEU",  # L
    "MET",  # M
    "ASN",  # N
    "PRO",  # P
    "GLN",  # Q
    "ARG",  # R
    "SER",  # S
    "THR",  # T
    "VAL",  # V
    "TRP",  # W
    "TYR",  # Y
    "UNK",  # X
)

VOCAB_SIZE: int = len(TOKEN_ORDER)  # 21

token_to_idx: dict[str, int] = {tok: i for i, tok in enumerate(TOKEN_ORDER)}
idx_to_token: dict[int, str] = {i: tok for i, tok in enumerate(TOKEN_ORDER)}

# ---------------------------------------------------------------------------
# Atom ordering (37-atom representation)
# ---------------------------------------------------------------------------

ATOM_ORDER: tuple[str, ...] = (
    "N",
    "CA",
    "C",
    "O",
    "CB",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
)

NUM_ATOMS_37: int = len(ATOM_ORDER)  # 37

atom_to_idx: dict[str, int] = {atom: i for i, atom in enumerate(ATOM_ORDER)}

# Backbone atom indices within the 37-atom representation
BACKBONE_ATOM_INDICES: tuple[int, ...] = (0, 1, 2, 3)  # N, CA, C, O
NUM_BACKBONE_ATOMS: int = 4

# 5 atoms used for RBF edge features: N, CA, C, O, virtual CB
BACKBONE_ATOMS: tuple[str, ...] = ("N", "CA", "C", "O", "CB")
NUM_RBF_ATOMS: int = 5
NUM_RBF_PAIRS: int = NUM_RBF_ATOMS**2  # 25

# Side-chain atom names for LigandMPNN atomization (indices 4–35 in ATOM_ORDER)
# Foundry uses 32 side-chain atoms (CB through NZ, excluding OXT)
SIDE_CHAIN_ATOM_NAMES: tuple[str, ...] = ATOM_ORDER[4:36]  # CB through NZ
NUM_SIDE_CHAIN_ATOMS: int = len(SIDE_CHAIN_ATOM_NAMES)  # 32

# ---------------------------------------------------------------------------
# Permutation utilities for legacy ↔ current conversion
# ---------------------------------------------------------------------------


def legacy_to_current_token_permutation() -> list[int]:
    """Return index mapping P such that ``current[i] = legacy[P[i]]``.

    Apply as: ``new_weight = old_weight[P]`` along the token dimension.
    """
    legacy_idx = {tok: i for i, tok in enumerate(LEGACY_TOKEN_ORDER)}
    return [legacy_idx[tok] for tok in TOKEN_ORDER]


def current_to_legacy_token_permutation() -> list[int]:
    """Return index mapping P such that ``legacy[i] = current[P[i]]``.

    Apply as: ``legacy_weight = current_weight[P]`` along the token dimension.
    """
    current_idx = {tok: i for i, tok in enumerate(TOKEN_ORDER)}
    return [current_idx[tok] for tok in LEGACY_TOKEN_ORDER]


def _enumerate_pairs_current() -> list[tuple[str, str]]:
    """Current (Foundry) outer-product ordering: N, CA, C, O, CB."""
    atoms = BACKBONE_ATOMS
    return [(a, b) for a in atoms for b in atoms]


# The exact legacy pair ordering from the dauparas ProteinMPNN/LigandMPNN code,
# as documented in Foundry's ``mpnn.utils.weights.load_legacy_weights``.
_LEGACY_PAIR_ORDER: list[tuple[str, str]] = [
    ("CA", "CA"),  # 0
    ("N", "N"),  # 1
    ("C", "C"),  # 2
    ("O", "O"),  # 3
    ("CB", "CB"),  # 4
    ("CA", "N"),  # 5
    ("CA", "C"),  # 6
    ("CA", "O"),  # 7
    ("CA", "CB"),  # 8
    ("N", "C"),  # 9
    ("N", "O"),  # 10
    ("N", "CB"),  # 11
    ("CB", "C"),  # 12
    ("CB", "O"),  # 13
    ("O", "C"),  # 14
    ("N", "CA"),  # 15
    ("C", "CA"),  # 16
    ("O", "CA"),  # 17
    ("CB", "CA"),  # 18
    ("C", "N"),  # 19
    ("O", "N"),  # 20
    ("CB", "N"),  # 21
    ("C", "CB"),  # 22
    ("O", "CB"),  # 23
    ("C", "O"),  # 24
]


def legacy_to_current_rbf_permutation() -> list[int]:
    """Return index mapping P such that ``current_rbf[i] = legacy_rbf[P[i]]``.

    Legacy format uses the dauparas pair ordering (see ``_LEGACY_PAIR_ORDER``).
    Current format uses standard outer-product ordering over (N, CA, C, O, CB).

    Each pair index covers 16 RBF kernels, so to reorder the full 400-dim
    feature vector, expand each pair index into a block of 16.
    """
    legacy_pair_idx = {pair: i for i, pair in enumerate(_LEGACY_PAIR_ORDER)}
    current_pairs = _enumerate_pairs_current()
    return [legacy_pair_idx[pair] for pair in current_pairs]


def current_to_legacy_rbf_permutation() -> list[int]:
    """Return index mapping P such that ``legacy_rbf[i] = current_rbf[P[i]]``."""
    current_pairs = _enumerate_pairs_current()
    current_pair_idx = {pair: i for i, pair in enumerate(current_pairs)}
    return [current_pair_idx[pair] for pair in _LEGACY_PAIR_ORDER]


def expand_pair_permutation(pair_perm: list[int], num_rbf: int = 16) -> list[int]:
    """Expand a 25-element pair permutation to a 400-element RBF feature permutation.

    Each pair index ``p`` covers RBF feature indices ``[p*num_rbf, (p+1)*num_rbf)``.
    """
    expanded: list[int] = []
    for p in pair_perm:
        expanded.extend(range(p * num_rbf, (p + 1) * num_rbf))
    return expanded
