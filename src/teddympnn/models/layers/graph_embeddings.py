"""Graph featurization modules for protein structures.

``ProteinFeatures`` builds a k-NN graph over CA atoms and computes RBF
distance features + positional encodings for all backbone atom pairs.
"""

from __future__ import annotations

import torch
from torch import nn

from teddympnn.models.layers.message_passing import gather_nodes
from teddympnn.models.layers.positional_encoding import PositionalEncodings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Virtual CB computation coefficients (from N, CA, C geometry)
_CB_A: float = -0.58273431
_CB_B: float = 0.56802827
_CB_C: float = -0.54067466

# RBF parameters
NUM_RBF: int = 16
RBF_D_MIN: float = 2.0
RBF_D_MAX: float = 22.0

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def compute_virtual_cb(X: torch.Tensor) -> torch.Tensor:
    """Compute virtual CB position from backbone N, CA, C coordinates.

    Args:
        X: Backbone coordinates, shape ``(..., 4, 3)`` with atoms [N, CA, C, O].

    Returns:
        Virtual CB coordinates, shape ``(..., 1, 3)``.
    """
    N = X[..., 0, :]
    CA = X[..., 1, :]
    C = X[..., 2, :]
    b = CA - N
    c = C - CA
    a = torch.cross(b, c, dim=-1)
    cb = _CB_A * a + _CB_B * b + _CB_C * c + CA
    return cb.unsqueeze(-2)


def rbf_encode(
    D: torch.Tensor, D_min: float = RBF_D_MIN, D_max: float = RBF_D_MAX, num_rbf: int = NUM_RBF
) -> torch.Tensor:
    """Gaussian RBF encoding of distances.

    Args:
        D: Distance tensor, arbitrary shape.
        D_min: Minimum center value.
        D_max: Maximum center value.
        num_rbf: Number of RBF kernels.

    Returns:
        RBF features, shape ``(*D.shape, num_rbf)``.
    """
    mu = torch.linspace(D_min, D_max, num_rbf, device=D.device, dtype=D.dtype)
    # Reshape mu for broadcasting: (1, 1, ..., num_rbf)
    shape = [1] * D.dim() + [num_rbf]
    mu = mu.view(*shape)
    # Gaussian width based on spacing
    sigma = (D_max - D_min) / num_rbf
    return torch.exp(-((D.unsqueeze(-1) - mu) ** 2) / sigma**2)


def compute_knn(
    coords: torch.Tensor,
    mask: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Compute k-nearest neighbor indices from coordinate distances.

    Args:
        coords: Representative atom coordinates, shape ``(B, L, 3)``.
        mask: Residue validity mask, shape ``(B, L)``.
        k: Number of neighbors.

    Returns:
        Neighbor indices, shape ``(B, L, K)``.
    """
    L = coords.shape[1]
    k = min(k, L)

    # Pairwise squared distances: (B, L, L)
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, L, L, 3)
    dist_sq = (diff**2).sum(dim=-1)  # (B, L, L)

    # Mask invalid residues with large distance
    inv_mask = ~mask.bool()
    dist_sq = dist_sq + inv_mask.unsqueeze(1).float() * 1e6

    # Top-k nearest (smallest distances)
    _, E_idx = dist_sq.topk(k, dim=-1, largest=False)
    assert isinstance(E_idx, torch.Tensor)
    return E_idx


# ---------------------------------------------------------------------------
# ProteinFeatures
# ---------------------------------------------------------------------------


class ProteinFeatures(nn.Module):
    """Graph featurization for protein backbone structures.

    Builds a k-NN graph from CA-CA distances and computes edge features from
    25 atom-pair RBF distance encodings (5 atoms × 5 atoms × 16 RBF) plus
    relative positional encodings.

    Foundry attribute names: ``positional_embedding``, ``edge_embedding``,
    ``edge_norm``.

    Args:
        num_positional_embeddings: Positional encoding output dim (default 16).
        num_rbf: Number of RBF kernels (default 16).
        top_k: Number of nearest neighbors (default 48).
        hidden_dim: Edge feature output dimensionality (default 128).
        max_relative_feature: Max relative position offset (default 32).
        dropout: Dropout probability (default 0.1).
    """

    def __init__(
        self,
        num_positional_embeddings: int = 16,
        num_rbf: int = NUM_RBF,
        top_k: int = 48,
        hidden_dim: int = 128,
        max_relative_feature: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_positional_embeddings = num_positional_embeddings
        self.num_rbf = num_rbf
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        # 25 atom pairs × num_rbf = 400 RBF features + positional embeddings
        num_edge_input = 25 * num_rbf + num_positional_embeddings  # 416

        self.positional_embedding = PositionalEncodings(
            num_positional_embeddings=num_positional_embeddings,
            max_relative_feature=max_relative_feature,
        )
        self.edge_embedding = nn.Linear(num_edge_input, hidden_dim, bias=False)
        self.edge_norm = nn.LayerNorm(hidden_dim)

    def _compute_rbf_features(
        self,
        atoms_5: torch.Tensor,
        E_idx: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Compute pairwise RBF distance features for all 25 atom pairs.

        Matches Foundry's vectorized approach: gather full neighbor coordinates,
        compute all pairwise distances at once with epsilon for numerical
        stability, then RBF encode.

        Args:
            atoms_5: 5-atom coordinates [N, CA, C, O, CB], shape ``(B, L, 5, 3)``.
            E_idx: Neighbor indices, shape ``(B, L, K)``.
            eps: Small constant added inside sqrt for stability.

        Returns:
            RBF features, shape ``(B, L, K, 400)``.
        """
        B, L, K = E_idx.shape
        num_atoms = atoms_5.shape[2]  # 5

        # Gather neighbor coordinates: (B, L, 5, 3) → flatten → gather → reshape
        X_flat = atoms_5.reshape(B, L, -1)  # (B, L, 15)
        X_flat_g = gather_nodes(X_flat, E_idx)  # (B, L, K, 15)
        X_g = X_flat_g.reshape(B, L, K, num_atoms, 3)  # (B, L, K, 5, 3)

        # Pairwise distances: (B, L, K, 5, 5) with eps for stability
        D = torch.sqrt(
            torch.sum(
                (atoms_5[:, :, None, :, None, :] - X_g[:, :, :, None, :, :]) ** 2,
                dim=-1,
            )
            + eps
        )

        # RBF encode all pairs at once: (B, L, K, 5, 5, num_rbf)
        RBF_all = rbf_encode(D, num_rbf=self.num_rbf)

        # Flatten: (B, L, K, 5*5*num_rbf) = (B, L, K, 400)
        return RBF_all.reshape(B, L, K, -1)

    def forward(
        self,
        X: torch.Tensor,
        residue_mask: torch.Tensor,
        R_idx: torch.Tensor,
        chain_labels: torch.Tensor,
        structure_noise: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Compute graph features from backbone coordinates.

        Args:
            X: Backbone coordinates [N, CA, C, O], shape ``(B, L, 4, 3)``.
            residue_mask: Residue validity mask, shape ``(B, L)``.
            R_idx: Residue indices, shape ``(B, L)``.
            chain_labels: Chain identifiers, shape ``(B, L)``.
            structure_noise: Gaussian noise std dev added to coordinates.

        Returns:
            Dict with ``E_idx`` (B, L, K) and ``E`` (B, L, K, hidden_dim).
        """
        # Add coordinate noise during training
        if structure_noise > 0.0 and self.training:
            X = X + torch.randn_like(X) * structure_noise

        # Compute virtual CB: (B, L, 1, 3)
        CB = compute_virtual_cb(X)
        # 5 atoms: [N, CA, C, O, CB] → (B, L, 5, 3)
        atoms_5 = torch.cat([X, CB], dim=-2)

        # k-NN from CA coordinates
        CA = X[:, :, 1, :]  # (B, L, 3)
        E_idx = compute_knn(CA, residue_mask, self.top_k)  # (B, L, K)

        # RBF features: (B, L, K, 400)
        rbf_features = self._compute_rbf_features(atoms_5, E_idx)

        # Positional encodings: (B, L, K, 16)
        pos_enc = self.positional_embedding(R_idx, chain_labels, E_idx)

        # Concatenate and project: (B, L, K, 416) → (B, L, K, 128)
        edge_input = torch.cat([pos_enc, rbf_features], dim=-1)
        E = self.edge_norm(self.edge_embedding(edge_input))

        return {"E_idx": E_idx, "E": E}
