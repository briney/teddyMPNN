"""Unified PPI dataset with partner-design view expansion.

``PPIDataset`` loads structures from any of the three data sources (teddymer,
NVIDIA complexes, PDB experimental) and expands each two-partner complex into
two partner-design training views.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from teddympnn.data.features import (
    derive_backbone,
    extract_ligand_atoms,
    extract_sidechain_atoms,
    parse_structure,
)

logger = logging.getLogger(__name__)

# UNK token index for padding
_UNK_IDX = 20  # matches token_to_idx["UNK"]


class PPIDataset(Dataset[dict[str, Any]]):
    """PyTorch Dataset for protein-protein interaction structures.

    Loads multi-chain structures from a manifest TSV and expands each
    two-partner complex into partner-design training views. Each view
    designates one partner for sequence prediction while conditioning
    on the other.

    The manifest TSV must contain columns:

    - ``structure_path``: path to PDB/mmCIF file
    - ``chain_A``: chain ID(s) for partner A (comma-separated if multiple)
    - ``chain_B``: chain ID(s) for partner B
    - ``source``: dataset source name (optional, for logging)

    Args:
        manifest_path: Path to manifest TSV file.
        cache_dir: Optional directory for caching parsed features as ``.pt``.
        max_residues: Skip structures with more residues than this.
        min_interface_contacts: Skip structures with fewer interface residues.
        include_ligand_atoms: Extract non-protein atoms for LigandMPNN.
        atomize_partner_sidechains: Include fixed-partner side-chain atoms
            in the ligand context (Y/Y_m/Y_t).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        cache_dir: str | Path | None = None,
        max_residues: int = 6000,
        min_interface_contacts: int = 4,
        include_ligand_atoms: bool = False,
        atomize_partner_sidechains: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_residues = max_residues
        self.min_interface_contacts = min_interface_contacts
        self.include_ligand_atoms = include_ligand_atoms
        self.atomize_partner_sidechains = atomize_partner_sidechains

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.manifest = pd.read_csv(self.manifest_path, sep="\t")

        # Build view index: each structure yields two views
        # (design A conditioned on B, design B conditioned on A)
        self._views: list[tuple[int, str, str]] = []
        self._num_residues: list[int] = []
        self._build_view_index()

    def _build_view_index(self) -> None:
        """Populate the view index from the manifest."""
        skipped_length = 0
        for idx, row in self.manifest.iterrows():
            structure_path = Path(row["structure_path"])
            if not structure_path.exists():
                logger.debug("Skipping missing structure: %s", structure_path)
                continue

            # Get residue count (from cache or quick parse)
            n_res = self._get_num_residues(int(idx), structure_path)
            if n_res > self.max_residues:
                skipped_length += 1
                continue

            chain_a = str(row["chain_A"])
            chain_b = str(row["chain_B"])

            # View 1: design A, condition on B
            self._views.append((int(idx), chain_a, chain_b))
            self._num_residues.append(n_res)

            # View 2: design B, condition on A
            self._views.append((int(idx), chain_b, chain_a))
            self._num_residues.append(n_res)

        if skipped_length > 0:
            logger.info(
                "Skipped %d structures exceeding max_residues=%d",
                skipped_length,
                self.max_residues,
            )
        logger.info(
            "Built %d partner-design views from %d structures",
            len(self._views),
            len(self.manifest),
        )

    def _get_num_residues(self, manifest_idx: int, structure_path: Path) -> int:
        """Get residue count, using cache if available."""
        if self.cache_dir:
            cache_path = self.cache_dir / f"{manifest_idx}_meta.pt"
            if cache_path.exists():
                meta = torch.load(cache_path, weights_only=True)
                return int(meta["num_residues"])

        # Quick parse to count residues
        try:
            parsed = parse_structure(structure_path)
            n_res = len(parsed["S"])
            if self.cache_dir:
                cache_path = self.cache_dir / f"{manifest_idx}_meta.pt"
                torch.save({"num_residues": n_res}, cache_path)
            return n_res
        except Exception:
            logger.debug("Failed to parse %s for residue count", structure_path)
            return 0

    def _load_features(self, manifest_idx: int) -> dict[str, Any]:
        """Load or compute features for a structure."""
        if self.cache_dir:
            cache_path = self.cache_dir / f"{manifest_idx}.pt"
            if cache_path.exists():
                return dict(torch.load(cache_path, weights_only=False))

        row = self.manifest.iloc[manifest_idx]
        structure_path = Path(row["structure_path"])
        features = parse_structure(structure_path)

        # Add backbone tensors
        X, X_m = derive_backbone(features["xyz_37"], features["xyz_37_m"])
        features["X"] = X
        features["X_m"] = X_m

        # Optionally extract ligand atoms
        if self.include_ligand_atoms:
            ligand = extract_ligand_atoms(structure_path)
            features.update(ligand)

        if self.cache_dir:
            cache_path = self.cache_dir / f"{manifest_idx}.pt"
            torch.save(features, cache_path)

        return features

    def __len__(self) -> int:
        return len(self._views)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        manifest_idx, design_chains, fixed_chains = self._views[idx]
        features = self._load_features(manifest_idx)

        chain_ids: list[str] = features["chain_ids"]
        design_chain_set = set(design_chains.split(","))
        fixed_chain_set = set(fixed_chains.split(","))

        L = len(chain_ids)

        # Build partner masks
        designed_residue_mask = torch.zeros(L, dtype=torch.bool)
        fixed_residue_mask = torch.zeros(L, dtype=torch.bool)
        for i, cid in enumerate(chain_ids):
            if cid in design_chain_set:
                designed_residue_mask[i] = True
            if cid in fixed_chain_set:
                fixed_residue_mask[i] = True

        result: dict[str, Any] = {
            "xyz_37": features["xyz_37"],
            "xyz_37_m": features["xyz_37_m"],
            "X": features["X"],
            "X_m": features["X_m"],
            "S": features["S"],
            "R_idx": features["R_idx"],
            "chain_labels": features["chain_labels"],
            "residue_mask": features["residue_mask"],
            "designed_residue_mask": designed_residue_mask,
            "fixed_residue_mask": fixed_residue_mask,
            "num_residues": L,
        }

        # Ligand context
        if self.include_ligand_atoms and "Y" in features:
            Y = features["Y"]
            Y_m = features["Y_m"]
            Y_t = features["Y_t"]

            # Optionally add fixed-partner side-chain atoms
            if self.atomize_partner_sidechains:
                sc = extract_sidechain_atoms(
                    features["xyz_37"],
                    features["xyz_37_m"],
                    features["S"],
                    fixed_residue_mask,
                )
                if sc["Y"].shape[0] > 0:
                    Y = torch.cat([Y, sc["Y"]], dim=0)
                    Y_m = torch.cat([Y_m, sc["Y_m"]], dim=0)
                    Y_t = torch.cat([Y_t, sc["Y_t"]], dim=0)

            result["Y"] = Y
            result["Y_m"] = Y_m
            result["Y_t"] = Y_t
        else:
            result["Y"] = torch.zeros(0, 3, dtype=torch.float32)
            result["Y_m"] = torch.zeros(0, dtype=torch.bool)
            result["Y_t"] = torch.zeros(0, dtype=torch.long)

        return result

    @property
    def lengths(self) -> list[int]:
        """Per-view residue counts for token-budget batch sampling."""
        return self._num_residues


class MixedDataLoader:
    """Wraps multiple PPIDatasets with configurable sampling weights.

    Yields batches drawn from constituent datasets proportionally to
    their weights, using a shared collator and sampler strategy.

    Args:
        datasets: List of PPIDatasets.
        weights: Sampling weight for each dataset (normalized internally).
        token_budget: Maximum total residues per batch.
        num_workers: DataLoader worker count.
        collate_fn: Collation function (typically ``PaddingCollator``).
        shuffle: Shuffle within each epoch.
    """

    def __init__(
        self,
        datasets: list[PPIDataset],
        weights: list[float],
        *,
        token_budget: int = 10_000,
        num_workers: int = 4,
        collate_fn: Any = None,
        shuffle: bool = True,
    ) -> None:
        if len(datasets) != len(weights):
            msg = f"Got {len(datasets)} datasets but {len(weights)} weights"
            raise ValueError(msg)

        self.datasets = datasets
        total = sum(weights)
        self.weights = [w / total for w in weights]
        self.token_budget = token_budget
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.shuffle = shuffle

        # Combine all datasets into a single ConcatDataset-like structure
        # with weighted sampling
        self._build_sampler()

    def _build_sampler(self) -> None:
        """Build a weighted random sampler across all datasets."""
        from torch.utils.data import ConcatDataset, WeightedRandomSampler

        from teddympnn.data.sampler import TokenBudgetBatchSampler

        self._concat: ConcatDataset[dict[str, Any]] = ConcatDataset(self.datasets)

        # Assign per-sample weights based on dataset membership
        sample_weights: list[float] = []
        all_lengths: list[int] = []
        for ds, w in zip(self.datasets, self.weights, strict=True):
            n = len(ds)
            per_sample = w / max(n, 1)
            sample_weights.extend([per_sample] * n)
            all_lengths.extend(ds.lengths)

        self._sample_weights = sample_weights
        self._all_lengths = all_lengths

        # Weighted random sampler for ordering
        self._sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=False,
        )

        # Token-budget batch sampler wrapping the weighted sampler
        self._batch_sampler = TokenBudgetBatchSampler(
            lengths=all_lengths,
            token_budget=self.token_budget,
            shuffle=self.shuffle,
        )

    def __iter__(self) -> Any:  # noqa: ANN204
        """Iterate over batches."""
        loader = DataLoader(
            self._concat,
            batch_sampler=self._batch_sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )
        yield from loader

    def __len__(self) -> int:
        return len(self._batch_sampler)
