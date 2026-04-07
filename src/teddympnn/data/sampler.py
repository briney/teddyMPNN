"""Token-budget batch sampler for variable-length protein structures.

Groups dataset indices by cumulative residue count rather than a fixed
batch size, keeping GPU memory usage roughly constant across batches
despite the wide length distribution of protein structures.
"""

from __future__ import annotations

import random
from collections.abc import Iterator  # noqa: TC003 — used in return type

from torch.utils.data import Sampler


class TokenBudgetBatchSampler(Sampler[list[int]]):
    """Batch sampler that groups examples by total residue count.

    Accumulates dataset indices until the total residue count reaches the
    token budget, then yields that group as a batch. Structures that
    individually exceed the budget are placed in singleton batches.

    Args:
        lengths: Per-example residue counts, indexed by dataset position.
        token_budget: Maximum total residues per batch.
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
        batch_tokens = 0

        for idx in indices:
            n = self.lengths[idx]

            # If adding this example would exceed budget, yield current batch
            if batch and batch_tokens + n > self.token_budget:
                yield batch
                batch = []
                batch_tokens = 0

            batch.append(idx)
            batch_tokens += n

        # Yield final batch
        if batch and not self.drop_last:
            yield batch

        self._epoch += 1

    def __len__(self) -> int:
        """Estimate number of batches (may vary across epochs with shuffling)."""
        if not self.lengths:
            return 0

        total_tokens = sum(self.lengths)
        # Approximate: actual count depends on packing order
        return max(1, (total_tokens + self.token_budget - 1) // self.token_budget)

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling (DDP compatibility)."""
        self._epoch = epoch
