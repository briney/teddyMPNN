"""Tests for PositionalEncodings."""

from __future__ import annotations

import torch

from teddympnn.models.layers.positional_encoding import PositionalEncodings


class TestPositionalEncodings:
    def test_output_shape(self) -> None:
        pe = PositionalEncodings(num_positional_embeddings=16, max_relative_feature=32)
        B, L, K = 2, 20, 10
        R_idx = torch.arange(L).unsqueeze(0).expand(B, -1)
        chain_labels = torch.zeros(B, L, dtype=torch.long)
        E_idx = torch.randint(0, L, (B, L, K))
        out = pe(R_idx, chain_labels, E_idx)
        assert out.shape == (B, L, K, 16)

    def test_num_positional_features(self) -> None:
        pe = PositionalEncodings(max_relative_feature=32)
        assert pe.num_positional_features == 66

    def test_same_chain_vs_cross_chain(self) -> None:
        """Inter-chain pairs should produce different encodings than intra-chain."""
        pe = PositionalEncodings(num_positional_embeddings=16)
        B, L = 1, 10
        R_idx = torch.arange(L).unsqueeze(0)

        # Neighbors include cross-chain indices (5, 6, 7)
        E_idx = torch.tensor([[[5, 6, 7]] * L])

        # Same chain
        chain_same = torch.zeros(B, L, dtype=torch.long)
        out_same = pe(R_idx, chain_same, E_idx)

        # Different chains: positions 5+ on chain 1
        chain_diff = torch.zeros(B, L, dtype=torch.long)
        chain_diff[0, 5:] = 1
        out_diff = pe(R_idx, chain_diff, E_idx)

        # Position 0 looking at neighbors 5,6,7 — same-chain vs cross-chain
        assert not torch.allclose(out_same[0, 0], out_diff[0, 0])

    def test_attribute_name(self) -> None:
        pe = PositionalEncodings()
        assert hasattr(pe, "embed_positional_features")

    def test_gradient_flow(self) -> None:
        pe = PositionalEncodings()
        R_idx = torch.arange(10).unsqueeze(0)
        chain_labels = torch.zeros(1, 10, dtype=torch.long)
        E_idx = torch.randint(0, 10, (1, 10, 5))
        out = pe(R_idx, chain_labels, E_idx)
        out.sum().backward()
        assert pe.embed_positional_features.weight.grad is not None
