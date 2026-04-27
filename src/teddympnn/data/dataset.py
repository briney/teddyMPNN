"""Unified PPI dataset with partner-design view expansion.

``PPIDataset`` loads structures from any of the three data sources (teddymer,
NVIDIA complexes, PDB experimental) and expands each two-partner complex into
two partner-design training views.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from teddympnn.data.features import (
    derive_backbone,
    extract_ligand_atoms,
    extract_sidechain_atoms,
    identify_interface_residues,
    parse_structure,
)
from teddympnn.data.sampler import TokenBudgetBatchSampler

logger = logging.getLogger(__name__)

REQUIRED_MANIFEST_COLUMNS: tuple[str, ...] = ("structure_path", "chain_A", "chain_B", "source")


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
        sidechain_atomization_probability: float = 1.0,
        sidechain_atomization_per_residue_probability: float = 1.0,
        source_filter: str | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_residues = max_residues
        self.min_interface_contacts = min_interface_contacts
        self.include_ligand_atoms = include_ligand_atoms
        self.atomize_partner_sidechains = atomize_partner_sidechains
        self.sidechain_atomization_probability = sidechain_atomization_probability
        self.sidechain_atomization_per_residue_probability = (
            sidechain_atomization_per_residue_probability
        )
        self.source_filter = source_filter

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.manifest = pd.read_csv(self.manifest_path, sep="\t")
        self._validate_manifest()
        if self.source_filter is not None:
            self.manifest = self.manifest[
                self.manifest["source"] == self.source_filter
            ].reset_index(drop=True)

        # Build view index: each structure yields two views
        # (design A conditioned on B, design B conditioned on A)
        self._views: list[tuple[int, str, str]] = []
        self._num_residues: list[int] = []
        self._build_view_index()

    def _validate_manifest(self) -> None:
        """Validate the source manifest schema."""
        missing = [c for c in REQUIRED_MANIFEST_COLUMNS if c not in self.manifest.columns]
        if missing:
            msg = f"Manifest {self.manifest_path} missing required columns: {missing}"
            raise ValueError(msg)

    def _build_view_index(self) -> None:
        """Populate the view index from the manifest."""
        skipped_length = 0
        skipped_interface = 0
        skipped_partner_masks = 0
        for idx, row in self.manifest.iterrows():
            structure_path = Path(row["structure_path"])
            if not structure_path.exists():
                logger.debug("Skipping missing structure: %s", structure_path)
                continue

            metadata = self._get_structure_metadata(int(idx), structure_path, row)
            n_res = metadata["num_residues"]
            if n_res > self.max_residues:
                skipped_length += 1
                continue
            if metadata["interface_residues"] < self.min_interface_contacts:
                skipped_interface += 1
                continue

            chain_a = str(row["chain_A"])
            chain_b = str(row["chain_B"])
            chain_ids = metadata["chain_ids"]
            if not self._chains_present(chain_ids, chain_a) or not self._chains_present(
                chain_ids, chain_b
            ):
                skipped_partner_masks += 1
                continue

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
        if skipped_interface > 0:
            logger.info(
                "Skipped %d structures below min_interface_contacts=%d",
                skipped_interface,
                self.min_interface_contacts,
            )
        if skipped_partner_masks > 0:
            logger.info("Skipped %d structures with missing partner chains", skipped_partner_masks)
        logger.info(
            "Built %d partner-design views from %d structures",
            len(self._views),
            len(self.manifest),
        )

    @staticmethod
    def _split_chain_ids(chains: str) -> set[str]:
        """Split comma-separated chain labels into a set."""
        return {c.strip() for c in chains.split(",") if c.strip()}

    @classmethod
    def _chains_present(cls, chain_ids: list[str], chains: str) -> bool:
        """Return True if at least one requested chain appears in the parsed structure."""
        requested = cls._split_chain_ids(chains)
        return bool(requested) and bool(requested.intersection(chain_ids))

    def _get_structure_metadata(
        self,
        manifest_idx: int,
        structure_path: Path,
        row: pd.Series,
    ) -> dict[str, Any]:
        """Get residue/interface metadata, using cache when available."""
        if self.cache_dir:
            cache_path = self.cache_dir / f"{manifest_idx}_meta.pt"
            if cache_path.exists():
                meta = torch.load(cache_path, weights_only=True)
                return dict(meta)

        try:
            parsed = parse_structure(structure_path)
            if "interface_residues" in row and not pd.isna(row["interface_residues"]):
                n_iface = int(row["interface_residues"])
            else:
                interface = identify_interface_residues(
                    parsed["xyz_37"],
                    parsed["xyz_37_m"],
                    parsed["chain_labels"],
                )
                n_iface = int(interface.sum().item())
            metadata = {
                "num_residues": len(parsed["S"]),
                "interface_residues": n_iface,
                "chain_ids": parsed["chain_ids"],
            }
            if self.cache_dir:
                cache_path = self.cache_dir / f"{manifest_idx}_meta.pt"
                torch.save(metadata, cache_path)
            return metadata
        except Exception:
            logger.debug("Failed to parse %s for residue count", structure_path)
            return {"num_residues": 0, "interface_residues": 0, "chain_ids": []}

    def _get_num_residues(self, manifest_idx: int, structure_path: Path) -> int:
        """Get residue count, using cache if available."""
        row = self.manifest.iloc[manifest_idx]
        return int(self._get_structure_metadata(manifest_idx, structure_path, row)["num_residues"])

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
        source = str(self.manifest.iloc[manifest_idx].get("source", "unknown"))

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

        if not designed_residue_mask.any() or not fixed_residue_mask.any():
            msg = (
                f"Invalid partner view in {self.manifest_path}: design={design_chains}, "
                f"fixed={fixed_chains}"
            )
            raise ValueError(msg)

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
            "source": source,
        }

        # Ligand context
        if self.include_ligand_atoms and "Y" in features:
            Y = features["Y"]
            Y_m = features["Y_m"]
            Y_t = features["Y_t"]

            # Optionally add fixed-partner side-chain atoms
            if self.atomize_partner_sidechains:
                atomize_mask = fixed_residue_mask
                if self.sidechain_atomization_probability < 1.0:
                    if random.random() >= self.sidechain_atomization_probability:
                        atomize_mask = torch.zeros_like(fixed_residue_mask)
                    else:
                        keep = (
                            torch.rand_like(fixed_residue_mask.float())
                            < self.sidechain_atomization_per_residue_probability
                        )
                        atomize_mask = fixed_residue_mask & keep
                sc = extract_sidechain_atoms(
                    features["xyz_37"],
                    features["xyz_37_m"],
                    features["S"],
                    atomize_mask,
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
        weighted: bool = True,
        rank: int = 0,
        world_size: int = 1,
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
        self.weighted = weighted
        self.rank = rank
        self.world_size = world_size
        self._epoch = 0

        self._build_sampler()

    def _build_sampler(self) -> None:
        """Build one token-budget DataLoader per source dataset."""
        self._loaders: list[DataLoader[dict[str, Any]]] = []
        self._loader_lengths: list[int] = []
        active_weights: list[float] = []
        for ds, weight in zip(self.datasets, self.weights, strict=True):
            dataset: Dataset[dict[str, Any]] = ds
            lengths = ds.lengths
            if self.world_size > 1:
                indices = list(range(self.rank, len(ds), self.world_size))
                if not indices:
                    continue
                dataset = Subset(ds, indices)
                lengths = [ds.lengths[i] for i in indices]
            if not lengths:
                continue
            sampler = TokenBudgetBatchSampler(
                lengths=lengths,
                token_budget=self.token_budget,
                shuffle=self.shuffle,
            )
            loader: DataLoader[dict[str, Any]] = DataLoader(
                dataset,
                batch_sampler=sampler,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers,
            )
            self._loaders.append(loader)
            self._loader_lengths.append(len(sampler))
            active_weights.append(weight)
        if not self._loaders:
            msg = "No non-empty datasets available for loading"
            raise ValueError(msg)
        total = sum(active_weights)
        self._active_weights = [w / total for w in active_weights]

    def __iter__(self) -> Any:  # noqa: ANN204
        """Iterate over source-weighted token-budget batches."""
        if not self.weighted:
            for loader in self._loaders:
                yield from loader
            return

        # Seed the source-choice RNG from the current epoch + rank so each
        # rank sees a different (but deterministic) source sequence per epoch.
        rng = random.Random(self._epoch * 1_000_003 + self.rank)
        iterators = [iter(loader) for loader in self._loaders]
        yielded = 0
        target_batches = len(self)
        while yielded < target_batches:
            source_idx = rng.choices(range(len(self._loaders)), weights=self._active_weights, k=1)[
                0
            ]
            try:
                batch = next(iterators[source_idx])
            except StopIteration:
                iterators[source_idx] = iter(self._loaders[source_idx])
                batch = next(iterators[source_idx])
            yielded += 1
            yield batch

    def __len__(self) -> int:
        return sum(self._loader_lengths)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch on this loader and all underlying batch samplers.

        Required for DDP reproducibility: each rank's per-epoch shuffle and
        weighted source-choice sequence is derived from ``epoch``, so callers
        must invoke this at the top of every training epoch.
        """
        self._epoch = int(epoch)
        for loader in self._loaders:
            sampler = loader.batch_sampler
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self._epoch)
