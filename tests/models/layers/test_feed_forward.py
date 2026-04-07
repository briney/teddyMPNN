"""Tests for PositionWiseFeedForward."""

from __future__ import annotations

import torch

from teddympnn.models.layers.feed_forward import PositionWiseFeedForward


class TestPositionWiseFeedForward:
    def test_output_shape(self) -> None:
        ffn = PositionWiseFeedForward(128, 512)
        x = torch.randn(2, 10, 128)
        out = ffn(x)
        assert out.shape == (2, 10, 128)

    def test_attribute_names(self) -> None:
        """Foundry requires W_in, W_out, act."""
        ffn = PositionWiseFeedForward(128, 512)
        assert hasattr(ffn, "W_in")
        assert hasattr(ffn, "W_out")
        assert hasattr(ffn, "act")

    def test_expansion_dims(self) -> None:
        ffn = PositionWiseFeedForward(64, 256)
        assert ffn.W_in.in_features == 64
        assert ffn.W_in.out_features == 256
        assert ffn.W_out.in_features == 256
        assert ffn.W_out.out_features == 64

    def test_gradient_flow(self) -> None:
        ffn = PositionWiseFeedForward(32, 128)
        x = torch.randn(1, 5, 32, requires_grad=True)
        out = ffn(x)
        out.sum().backward()
        assert x.grad is not None
        for param in ffn.parameters():
            assert param.grad is not None
