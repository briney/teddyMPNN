"""Token-budget batch sampler for variable-length protein structures.

Bounds the *padded* compute per batch (``B * L_max``) instead of the sum of
unpadded residue counts, so a single long example cannot blow the batch up
past the configured budget once the collator pads everything to ``L_max``.
"""

from __future__ import annotations

import random
from collections.abc import Iterator  # noqa: TC003 — used in return type

from torch.utils.data import Sampler


class TokenBudgetBatchSampler(Sampler[list[int]]):
    """Batch sampler bounded by padded-token budget ``B * L_max``.

    For each candidate index, the sampler computes the prospective
    ``(len(batch) + 1) * max(L_so_far, n)`` cost — exactly the residue
    tensor footprint after padding to the new longest example. The batch
    is flushed before adding any index that would exceed the budget. A
    single example longer than the entire budget is emitted as a singleton
    batch.

    Args:
        lengths: Per-example residue counts, indexed by dataset position.
        token_budget: Maximum padded residues per batch (``B * L_max``).
            For LigandMPNN training the recommended value is ~6,000;
            ProteinMPNN tolerates ~10,000.
        shuffle: Randomize example order each epoch.
        drop_last: Drop the final incomplete batch.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        lengths: list[int],
        token_budget: int = 10_000,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        self.lengths = lengths
        self.token_budget = token_budget
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        indices = list(range(len(self.lengths)))

        if self.shuffle:
            rng = random.Random(self.seed + self._epoch)
            rng.shuffle(indices)

        batch: list[int] = []
        current_max = 0

        for idx in indices:
            n = self.lengths[idx]

            # Padded cost if we add this example: (B + 1) * max(current_max, n).
            prospective_max = max(current_max, n)
            prospective_cost = (len(batch) + 1) * prospective_max

            if batch and prospective_cost > self.token_budget:
                yield batch
                batch = []
                current_max = 0
                prospective_max = n

            batch.append(idx)
            current_max = prospective_max

        if batch and not self.drop_last:
            yield batch

        self._epoch += 1

    def __len__(self) -> int:
        """Estimate the number of batches.

        Approximates by sorting lengths descending and packing greedily —
        actual count varies with shuffle order but the estimate is a
        reasonable upper bound for ``DataLoader`` progress reporting.
        """
        if not self.lengths:
            return 0

        sorted_lengths = sorted(self.lengths, reverse=True)
        batches = 0
        batch_count = 0
        current_max = 0
        for n in sorted_lengths:
            prospective_max = max(current_max, n)
            prospective_cost = (batch_count + 1) * prospective_max
            if batch_count > 0 and prospective_cost > self.token_budget:
                batches += 1
                batch_count = 0
                current_max = 0
                prospective_max = n
            batch_count += 1
            current_max = prospective_max
        if batch_count > 0:
            batches += 1
        return max(1, batches)

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling (DDP compatibility)."""
        self._epoch = epoch
