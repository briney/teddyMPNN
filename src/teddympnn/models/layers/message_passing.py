"""Message passing layers for encoder and decoder.

Includes helper functions for gathering node/edge features at neighbor
indices and the ``EncLayer`` / ``DecLayer`` modules that compose the
ProteinMPNN encoder and decoder stacks.
"""

from __future__ import annotations

import torch
from torch import nn

from teddympnn.models.layers.feed_forward import PositionWiseFeedForward

# ---------------------------------------------------------------------------
# Gather / concatenation helpers
# ---------------------------------------------------------------------------


def gather_nodes(node_features: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """Gather node features at neighbor indices.

    Args:
        node_features: Node features, shape ``(B, L, H)``.
        neighbor_idx: Neighbor indices, shape ``(B, L, K)``.

    Returns:
        Gathered features, shape ``(B, L, K, H)``.
    """
    # neighbor_idx: (B, L, K) → (B, L, K, 1) → (B, L, K, H)
    idx_expanded = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, node_features.shape[-1])
    # Gather from dim=1 (L dimension)
    K = neighbor_idx.shape[-1]
    return torch.gather(node_features.unsqueeze(2).expand(-1, -1, K, -1), 1, idx_expanded)


def gather_edges(edge_features: torch.Tensor, neighbor_idx: torch.Tensor) -> torch.Tensor:
    """Gather edge features at neighbor indices.

    When edge features are stored as ``(B, L, L, H)`` (full pairwise), gather
    the K neighbors per node. This function handles the case where edge
    features are already in ``(B, L, K, H)`` format (returns as-is).

    Args:
        edge_features: Edge features, shape ``(B, L, L, H)`` or ``(B, L, K, H)``.
        neighbor_idx: Neighbor indices, shape ``(B, L, K)``.

    Returns:
        Gathered features, shape ``(B, L, K, H)``.
    """
    if edge_features.shape[2] == neighbor_idx.shape[2]:
        # Already in (B, L, K, H) format
        return edge_features

    B, L, _, H = edge_features.shape
    K = neighbor_idx.shape[2]
    # (B, L, K) → (B, L, K, H)
    idx_expanded = neighbor_idx.unsqueeze(-1).expand(B, L, K, H)
    return torch.gather(edge_features, 2, idx_expanded)


def cat_neighbors_nodes(
    h_nodes: torch.Tensor,
    h_edges: torch.Tensor,
    E_idx: torch.Tensor,
) -> torch.Tensor:
    """Concatenate edge features with gathered neighbor node features.

    Foundry convention: edges first, then nodes.

    Args:
        h_nodes: Node features, shape ``(B, L, H_node)``.
        h_edges: Edge features, shape ``(B, L, K, H_edge)``.
        E_idx: Neighbor indices, shape ``(B, L, K)``.

    Returns:
        Concatenated features, shape ``(B, L, K, H_edge + H_node)``.
    """
    h_neighbors = gather_nodes(h_nodes, E_idx)
    return torch.cat([h_edges, h_neighbors], dim=-1)


# ---------------------------------------------------------------------------
# Encoder layer
# ---------------------------------------------------------------------------


class EncLayer(nn.Module):
    """Encoder message passing layer with node and edge updates.

    Node update: gather neighbor node + edge features → concatenate with source
    node → 3-layer MLP → sum / scale → residual + LayerNorm → FFN → residual +
    LayerNorm.

    Edge update: same MLP pattern with W11/W12/W13 → residual + LayerNorm.

    Args:
        num_hidden: Hidden dimensionality (H).
        num_in: Input dimensionality for the MLP (typically ``3 * H``).
        dropout: Dropout probability.
        scale: Aggregation scale divisor.
    """

    def __init__(
        self,
        num_hidden: int,
        num_in: int,
        dropout: float = 0.1,
        scale: float = 30.0,
    ) -> None:
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale

        # Node update MLP
        self.W1 = nn.Linear(num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)

        # Edge update MLP
        self.W11 = nn.Linear(num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)

        # Norms and dropout
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Position-wise FFN (node only)
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

        self.act = nn.GELU()

    def forward(
        self,
        h_V: torch.Tensor,
        h_E: torch.Tensor,
        E_idx: torch.Tensor,
        mask_V: torch.Tensor,
        mask_E: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one encoder message passing step.

        Args:
            h_V: Node features, shape ``(B, L, H)``.
            h_E: Edge features, shape ``(B, L, K, H)``.
            E_idx: Neighbor indices, shape ``(B, L, K)``.
            mask_V: Node mask, shape ``(B, L)`` or ``(B, L, 1)``.
            mask_E: Edge mask, shape ``(B, L, K)`` or ``(B, L, K, 1)``.

        Returns:
            Updated (h_V, h_E) with same shapes.
        """
        # Ensure masks have trailing dim for broadcasting
        if mask_V.dim() == 2:
            mask_V = mask_V.unsqueeze(-1)  # (B, L, 1)
        if mask_E.dim() == 3:
            mask_E = mask_E.unsqueeze(-1)  # (B, L, K, 1)

        # --- Node update ---
        # cat(edge, neighbor_node) → (B, L, K, 2H)
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        # Prepend source node → (B, L, K, 3H)
        K = E_idx.shape[2]
        h_V_expand = h_V.unsqueeze(2).expand(-1, -1, K, -1)
        h_EV = torch.cat([h_V_expand, h_EV], dim=-1)
        # 3-layer MLP
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        # Masked sum aggregation with scale
        h_message = h_message * mask_E
        dh = h_message.sum(dim=2) / self.scale
        # Residual + norm + FFN + norm
        h_V = self.norm1(h_V + self.dropout1(dh))
        dff = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dff))
        h_V = h_V * mask_V

        # --- Edge update ---
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(2).expand(-1, -1, K, -1)
        h_EV = torch.cat([h_V_expand, h_EV], dim=-1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        h_E = h_E * mask_E

        return h_V, h_E


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------


class DecLayer(nn.Module):
    """Decoder message passing layer (node update only, no edge update).

    The caller provides pre-concatenated neighbor features with dimension
    ``num_in - H``. The layer internally prepends the source node ``h_V``
    to form the full ``num_in``-dimensional input to the MLP.

    ``num_in`` by context:
    - ProteinMPNN decoder: ``4H`` → caller passes 3H, layer adds source H
    - LigandMPNN protein-ligand context: ``3H`` → caller passes 2H
    - LigandMPNN ligand subgraph: ``2H`` → caller passes H

    Args:
        num_hidden: Hidden dimensionality (H).
        num_in: Full MLP input dimensionality (including source node H).
        dropout: Dropout probability.
        scale: Aggregation scale divisor.
    """

    def __init__(
        self,
        num_hidden: int,
        num_in: int,
        dropout: float = 0.1,
        scale: float = 30.0,
    ) -> None:
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale

        # Node update MLP
        self.W1 = nn.Linear(num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)

        # Norms and dropout
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Position-wise FFN
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

        self.act = nn.GELU()

    def forward(
        self,
        h_V: torch.Tensor,
        h_E: torch.Tensor,
        mask_V: torch.Tensor | None = None,
        mask_E: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one decoder message passing step.

        Args:
            h_V: Node features, shape ``(B, ..., L, H)``.
            h_E: Pre-concatenated neighbor features, shape
                ``(B, ..., L, K, num_in - H)``. The layer prepends ``h_V``.
            mask_V: Node mask, shape ``(B, ..., L)``.
            mask_E: Edge mask, shape ``(B, ..., L, K)``.

        Returns:
            Updated h_V, shape ``(B, ..., L, H)``.
        """
        # Prepend source node to neighbor features: (B,...,L,K,num_in)
        h_V_expand = h_V.unsqueeze(-2).expand(*h_V.shape[:-1], h_E.size(-2), h_V.shape[-1])
        h_EV = torch.cat([h_V_expand, h_E], dim=-1)

        # 3-layer MLP
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))

        # Masked sum aggregation with scale
        if mask_E is not None:
            h_message = mask_E.unsqueeze(-1) * h_message
        dh = h_message.sum(dim=-2) / self.scale

        # Residual + norm + FFN + norm
        h_V = self.norm1(h_V + self.dropout1(dh))
        dff = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dff))

        if mask_V is not None:
            h_V = mask_V.unsqueeze(-1) * h_V

        return h_V
