"""Position-wise feed-forward network."""

from __future__ import annotations

import torch
from torch import nn


class PositionWiseFeedForward(nn.Module):
    """Two-layer MLP with GELU activation.

    Matches Foundry attribute names: ``W_in``, ``W_out``, ``act``.

    Args:
        num_hidden: Input and output dimensionality.
        num_ff_hidden: Inner (expanded) dimensionality. Typically ``4 * num_hidden``.
    """

    def __init__(self, num_hidden: int, num_ff_hidden: int) -> None:
        super().__init__()
        self.W_in = nn.Linear(num_hidden, num_ff_hidden, bias=True)
        self.W_out = nn.Linear(num_ff_hidden, num_hidden, bias=True)
        self.act = nn.GELU()

    def forward(self, h_V: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network.

        Args:
            h_V: Input tensor of shape ``(..., num_hidden)``.

        Returns:
            Output tensor of same shape as input.
        """
        return self.W_out(self.act(self.W_in(h_V)))  # type: ignore[no-any-return]
