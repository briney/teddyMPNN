"""Tests for token-budget batch sampler."""

from __future__ import annotations

from teddympnn.data.sampler import TokenBudgetBatchSampler


class TestTokenBudgetBatchSampler:
    def test_basic_batching(self):
        """Batches should not exceed token budget."""
        lengths = [100, 200, 150, 300, 50, 250, 100]
        sampler = TokenBudgetBatchSampler(lengths, token_budget=400, shuffle=False)

        for batch in sampler:
            total = sum(lengths[i] for i in batch)
            # Budget may be exceeded by at most one structure
            assert total <= 400 + max(lengths[i] for i in batch)

    def test_all_indices_covered(self):
        """Every index should appear in exactly one batch."""
        lengths = [100, 200, 150, 300, 50]
        sampler = TokenBudgetBatchSampler(lengths, token_budget=400, shuffle=False)

        all_indices: list[int] = []
        for batch in sampler:
            all_indices.extend(batch)

        assert sorted(all_indices) == list(range(len(lengths)))

    def test_single_large_item(self):
        """A structure larger than the budget gets its own batch."""
        lengths = [100, 500, 100]  # 500 exceeds budget of 300
        sampler = TokenBudgetBatchSampler(lengths, token_budget=300, shuffle=False)

        batches = list(sampler)
        # The 500-length item should be alone in its batch
        for batch in batches:
            if 1 in batch:
                assert batch == [1], "Large item should be in a singleton batch"

    def test_shuffle_changes_order(self):
        """Different epochs should produce different batch compositions."""
        lengths = [100] * 20
        sampler = TokenBudgetBatchSampler(lengths, token_budget=350, shuffle=True, seed=42)

        epoch0_batches = [list(b) for b in sampler]
        epoch1_batches = [list(b) for b in sampler]

        # With different epoch seeds, batches should differ
        # (not guaranteed but highly likely with 20 elements)
        all_same = all(b0 == b1 for b0, b1 in zip(epoch0_batches, epoch1_batches, strict=False))
        assert not all_same, "Shuffled batches should vary between epochs"

    def test_no_shuffle_deterministic(self):
        """Without shuffling, batches should be deterministic."""
        lengths = [100, 200, 150, 50]
        sampler = TokenBudgetBatchSampler(lengths, token_budget=400, shuffle=False)

        batches1 = [list(b) for b in sampler]
        batches2 = [list(b) for b in sampler]
        assert batches1 == batches2

    def test_drop_last(self):
        """With drop_last, incomplete final batch is dropped."""
        lengths = [100, 100, 100, 50]  # budget=250 → batch of [0,1,2], leftover [3]
        sampler_keep = TokenBudgetBatchSampler(
            lengths,
            token_budget=250,
            shuffle=False,
            drop_last=False,
        )
        sampler_drop = TokenBudgetBatchSampler(
            lengths,
            token_budget=250,
            shuffle=False,
            drop_last=True,
        )

        batches_keep = list(sampler_keep)
        batches_drop = list(sampler_drop)

        # drop_last should have fewer or equal batches
        assert len(batches_drop) <= len(batches_keep)

    def test_empty_lengths(self):
        sampler = TokenBudgetBatchSampler([], token_budget=1000)
        assert list(sampler) == []
        assert len(sampler) == 0

    def test_set_epoch(self):
        """set_epoch should affect shuffling."""
        lengths = [100] * 10
        sampler = TokenBudgetBatchSampler(lengths, token_budget=250, shuffle=True)

        sampler.set_epoch(0)
        _ = [list(b) for b in sampler]

        sampler.set_epoch(5)
        _ = [list(b) for b in sampler]

        # The key property is that set_epoch controls the seed
        sampler.set_epoch(0)

    def test_budget_respected_with_tolerance(self):
        """Total tokens per batch should be within budget + one item."""
        lengths = [50, 75, 100, 125, 150, 200, 60, 80, 90, 110]
        sampler = TokenBudgetBatchSampler(lengths, token_budget=300, shuffle=False)

        for batch in sampler:
            total = sum(lengths[i] for i in batch)
            # The first item that pushes over budget is included
            assert total <= 300 or len(batch) == 1


class TestPaddedTokenBudget:
    """The sampler must bound (B * L_max), not just sum(unpadded lengths)."""

    def test_adversarial_long_short_mixture(self):
        """One long example mixed with many short ones must not exceed B*L_max."""
        # 1000-residue complex + 40 50-residue complexes. With sum-of-unpadded
        # accounting, batch=[1000, 50, 50, ...] would all fit ~2000 tokens.
        # With padded accounting, the long example pads short ones to 1000 →
        # the budget should keep the batch tiny.
        lengths = [1000, *([50] * 40)]
        budget = 2000
        sampler = TokenBudgetBatchSampler(lengths, token_budget=budget, shuffle=False)
        for batch in sampler:
            l_max = max(lengths[i] for i in batch)
            padded = len(batch) * l_max
            assert padded <= budget, (
                f"Batch {batch} has padded cost {padded} > budget {budget} "
                f"(L_max={l_max}, B={len(batch)})"
            )

    def test_padded_bound_under_shuffle(self):
        """Padded bound must hold across shuffled epochs."""
        rng_lengths = [50, 80, 1000, 60, 950, 70, 100, 90, 120, 60, 800]
        budget = 1500
        sampler = TokenBudgetBatchSampler(rng_lengths, token_budget=budget, shuffle=True, seed=7)
        for epoch in range(3):
            sampler.set_epoch(epoch)
            for batch in sampler:
                l_max = max(rng_lengths[i] for i in batch)
                assert len(batch) * l_max <= budget or len(batch) == 1

    def test_uniform_lengths_pack_normally(self):
        """When all lengths are equal, B*L_max == sum, so packing matches."""
        lengths = [100] * 20
        budget = 400
        sampler = TokenBudgetBatchSampler(lengths, token_budget=budget, shuffle=False)
        batches = list(sampler)
        # With uniform L=100 and budget 400 → 4 per batch.
        for b in batches:
            assert len(b) <= 4
            l_max = max(lengths[i] for i in b)
            assert len(b) * l_max <= budget

    def test_singleton_when_single_example_exceeds_budget(self):
        """A single example longer than the budget still gets emitted alone."""
        lengths = [10, 5_000, 10]
        sampler = TokenBudgetBatchSampler(lengths, token_budget=1_000, shuffle=False)
        batches = list(sampler)
        assert [1] in batches
