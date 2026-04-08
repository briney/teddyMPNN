"""Data pipeline for teddyMPNN training.

Provides structure parsing, dataset acquisition, and data loading utilities
for protein-protein interaction fine-tuning.
"""

from __future__ import annotations

from teddympnn.data.collator import PaddingCollator
from teddympnn.data.dataset import PPIDataset
from teddympnn.data.features import (
    derive_backbone,
    extract_ligand_atoms,
    identify_interface_residues,
    parse_structure,
)
from teddympnn.data.sampler import TokenBudgetBatchSampler
from teddympnn.data.splits import prepare_manifests

__all__ = [
    "PaddingCollator",
    "PPIDataset",
    "TokenBudgetBatchSampler",
    "derive_backbone",
    "extract_ligand_atoms",
    "identify_interface_residues",
    "parse_structure",
    "prepare_manifests",
]
