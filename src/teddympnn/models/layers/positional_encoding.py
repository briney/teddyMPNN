"""Relative positional encoding for protein graphs."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PositionalEncodings(nn.Module):
    """Relative residue positional encoding with inter-chain bucket.

    Computes relative residue offsets, clips to ``[-max_relative_feature,
    +max_relative_feature]``, one-hot encodes (with a separate inter-chain
    bucket), and projects to ``num_positional_embeddings`` dimensions.

    Total one-hot classes = ``2 * max_relative_feature + 1 + 1``:
    - 65 intra-chain offset bins (``-32..+32``)
    - 1 inter-chain bucket (index 65)

    Foundry attribute name: ``embed_positional_features``.

    Args:
        num_positional_embeddings: Output dimensionality (default 16).
        max_relative_feature: Maximum relative offset magnitude (default 32).
    """

    def __init__(
        self,
        num_positional_embeddings: int = 16,
        max_relative_feature: int = 32,
    ) -> None:
        super().__init__()
        self.num_positional_embeddings = num_positional_embeddings
        self.max_relative_feature = max_relative_feature
        self.num_positional_features = 2 * max_relative_feature + 1 + 1  # 66

        self.embed_positional_features = nn.Linear(
            self.num_positional_features,
            num_positional_embeddings,
            bias=True,
        )

    def forward(
        self,
        R_idx: torch.Tensor,
        chain_labels: torch.Tensor,
        E_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute positional encodings for neighbor edges.

        Args:
            R_idx: Residue indices, shape ``(B, L)``.
            chain_labels: Chain identifiers, shape ``(B, L)``.
            E_idx: Neighbor indices, shape ``(B, L, K)``.

        Returns:
            Positional encodings, shape ``(B, L, K, num_positional_embeddings)``.
        """
        # Gather residue indices and chain labels at neighbor positions
        # R_idx: (B, L) → R_idx_neighbors: (B, L, K)
        R_idx_neighbors = torch.gather(
            R_idx.unsqueeze(-1).expand(-1, -1, E_idx.shape[-1]),
            dim=1,
            index=E_idx,
        )
        chain_neighbors = torch.gather(
            chain_labels.unsqueeze(-1).expand(-1, -1, E_idx.shape[-1]),
            dim=1,
            index=E_idx,
        )

        # Relative offsets: (B, L, K)
        d = R_idx.unsqueeze(-1) - R_idx_neighbors
        d = d.clamp(-self.max_relative_feature, self.max_relative_feature)
        # Shift to non-negative: [-32..32] → [0..64]
        d = d + self.max_relative_feature

        # Same-chain mask: (B, L, K)
        same_chain = chain_labels.unsqueeze(-1) == chain_neighbors

        # Inter-chain pairs get the last bucket index (65)
        d = torch.where(same_chain, d, torch.full_like(d, self.num_positional_features - 1))

        # One-hot encode: (B, L, K, 66)
        d_onehot = F.one_hot(d, num_classes=self.num_positional_features).float()

        # Project: (B, L, K, 16)
        return self.embed_positional_features(d_onehot)  # type: ignore[no-any-return]
