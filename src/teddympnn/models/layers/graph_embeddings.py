"""Graph featurization modules for protein and protein-ligand structures.

``ProteinFeatures`` builds a k-NN graph over CA atoms and computes RBF
distance features + positional encodings for all backbone atom pairs.

``ProteinFeaturesLigand`` extends this with a protein-to-ligand context graph
and an intraligand subgraph for LigandMPNN.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from teddympnn.models.layers.message_passing import gather_nodes
from teddympnn.models.layers.positional_encoding import PositionalEncodings
from teddympnn.models.tokens import SIDE_CHAIN_ATOM_NAMES

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

# Element type vocabulary for LigandMPNN (119 types in Foundry current format)
NUM_ELEMENT_TYPES: int = 119
NUM_PERIODIC_GROUPS: int = 19
NUM_PERIODIC_PERIODS: int = 8
ATOM_TYPE_EMBED_DIM: int = NUM_ELEMENT_TYPES + NUM_PERIODIC_GROUPS + NUM_PERIODIC_PERIODS  # 146
ATOM_TYPE_OUTPUT_DIM: int = 64

# Number of context (non-protein) atoms per residue in LigandMPNN
NUM_CONTEXT_ATOMS: int = 25

# Angle features: sin/cos of two angles = 4
NUM_ANGLE_FEATURES: int = 4

_ELEMENT_GROUPS: tuple[int, ...] = (
    1,
    18,
    1,
    2,
    13,
    14,
    15,
    16,
    17,
    18,
    1,
    2,
    13,
    14,
    15,
    16,
    17,
    18,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    1,
    2,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    1,
    2,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    0,
)
_ELEMENT_PERIODS: tuple[int, ...] = (
    1,
    1,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    6,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    7,
    0,
)
_ELEMENT_INDEX_BY_SYMBOL: dict[str, int] = {
    "C": 5,
    "N": 6,
    "O": 7,
    "S": 15,
}


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


def default_side_chain_atom_types() -> torch.Tensor:
    """Return element indices for the 32 side-chain atom slots."""
    return torch.tensor(
        [_ELEMENT_INDEX_BY_SYMBOL.get(name[0].upper(), NUM_ELEMENT_TYPES - 1) for name in SIDE_CHAIN_ATOM_NAMES],
        dtype=torch.long,
    )


def default_periodic_table_groups() -> torch.Tensor:
    """Return periodic-table group classes for the 119 atom-type vocabulary."""
    return torch.tensor(_ELEMENT_GROUPS, dtype=torch.long)


def default_periodic_table_periods() -> torch.Tensor:
    """Return periodic-table period classes for the 119 atom-type vocabulary."""
    return torch.tensor(_ELEMENT_PERIODS, dtype=torch.long)


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


# ---------------------------------------------------------------------------
# ProteinFeaturesLigand
# ---------------------------------------------------------------------------


class ProteinFeaturesLigand(ProteinFeatures):
    """Graph featurization for protein-ligand structures (LigandMPNN).

    Extends ``ProteinFeatures`` with three additional feature sets:

    1. **Protein-to-ligand edges** — RBF from 5 backbone atoms to each context
       atom + atom type embedding + angle features.
    2. **Ligand subgraph nodes** — Element type embeddings for non-protein atoms.
    3. **Ligand subgraph edges** — Inter-atom RBF distances within the ligand.

    Args:
        num_positional_embeddings: Positional encoding output dim.
        num_rbf: Number of RBF kernels.
        top_k: Protein backbone k-NN count (default 32 for LigandMPNN).
        hidden_dim: Feature output dimensionality.
        max_relative_feature: Max relative position offset.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        num_positional_embeddings: int = 16,
        num_rbf: int = NUM_RBF,
        top_k: int = 32,
        hidden_dim: int = 128,
        max_relative_feature: int = 32,
        dropout: float = 0.1,
        num_context_atoms: int = NUM_CONTEXT_ATOMS,
    ) -> None:
        super().__init__(
            num_positional_embeddings=num_positional_embeddings,
            num_rbf=num_rbf,
            top_k=top_k,
            hidden_dim=hidden_dim,
            max_relative_feature=max_relative_feature,
            dropout=dropout,
        )
        self.num_context_atoms = num_context_atoms

        # Atom type embedding: 146-dim input → 64-dim output
        self.embed_atom_type_features = nn.Linear(
            ATOM_TYPE_EMBED_DIM, ATOM_TYPE_OUTPUT_DIM, bias=True
        )

        # Protein-to-ligand edge features
        # 5 backbone atoms × num_rbf (80) + atom type embed (64) + angles (4) = 148
        num_node_input = 5 * num_rbf + ATOM_TYPE_OUTPUT_DIM + NUM_ANGLE_FEATURES  # 148
        self.node_embedding = nn.Linear(num_node_input, hidden_dim, bias=True)
        self.node_norm = nn.LayerNorm(hidden_dim)

        # Ligand subgraph node embedding
        self.ligand_subgraph_node_embedding = nn.Linear(ATOM_TYPE_EMBED_DIM, hidden_dim, bias=False)
        self.ligand_subgraph_node_norm = nn.LayerNorm(hidden_dim)

        # Ligand subgraph edge embedding
        self.ligand_subgraph_edge_embedding = nn.Linear(num_rbf, hidden_dim, bias=False)
        self.ligand_subgraph_edge_norm = nn.LayerNorm(hidden_dim)

        # Registered buffers (not learned, copied from model not checkpoint)
        # These are populated by the parent LigandMPNN module
        self.register_buffer(
            "side_chain_atom_types",
            default_side_chain_atom_types(),
        )
        self.register_buffer(
            "periodic_table_groups",
            default_periodic_table_groups(),
        )
        self.register_buffer(
            "periodic_table_periods",
            default_periodic_table_periods(),
        )

    def _encode_atom_type(self, atom_types: torch.Tensor) -> torch.Tensor:
        """Encode atom types as element + periodic table group + period features.

        Args:
            atom_types: Atom type indices, shape ``(...)``.

        Returns:
            Atom type features, shape ``(..., 146)``.
        """
        # One-hot element type: (..., 119)
        elem_onehot = F.one_hot(
            atom_types.clamp(0, NUM_ELEMENT_TYPES - 1), num_classes=NUM_ELEMENT_TYPES
        ).float()
        # Periodic table group: (..., 19)
        groups = self.periodic_table_groups[atom_types.clamp(0, NUM_ELEMENT_TYPES - 1)]
        group_onehot = F.one_hot(groups, num_classes=NUM_PERIODIC_GROUPS).float()
        # Periodic table period: (..., 8)
        periods = self.periodic_table_periods[atom_types.clamp(0, NUM_ELEMENT_TYPES - 1)]
        period_onehot = F.one_hot(periods, num_classes=NUM_PERIODIC_PERIODS).float()
        # Concatenate: (..., 146)
        return torch.cat([elem_onehot, group_onehot, period_onehot], dim=-1)

    def _compute_angle_features(
        self,
        X: torch.Tensor,
        Y_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Compute angle features between backbone and context atoms.

        Computes sin/cos of the angle at CA between N-CA-context and
        C-CA-context directions.

        Args:
            X: Backbone coordinates [N, CA, C, O], shape ``(B, L, 4, 3)``.
            Y_coords: Context atom coordinates, shape ``(B, L, Kc, 3)``.

        Returns:
            Angle features, shape ``(B, L, Kc, 4)``.
        """
        N = X[:, :, 0, :]  # (B, L, 3)
        CA = X[:, :, 1, :]  # (B, L, 3)
        C = X[:, :, 2, :]  # (B, L, 3)

        # Directions from CA
        v1 = F.normalize(N - CA, dim=-1).unsqueeze(2)  # (B, L, 1, 3)
        v2 = F.normalize(C - CA, dim=-1).unsqueeze(2)  # (B, L, 1, 3)
        vy = F.normalize(Y_coords - CA.unsqueeze(2), dim=-1)  # (B, L, Kc, 3)

        # cos/sin of angles via dot/cross products
        cos1 = (v1 * vy).sum(dim=-1)  # (B, L, Kc)
        cos2 = (v2 * vy).sum(dim=-1)
        sin1 = torch.cross(v1.expand_as(vy), vy, dim=-1).norm(dim=-1)
        sin2 = torch.cross(v2.expand_as(vy), vy, dim=-1).norm(dim=-1)

        return torch.stack([cos1, sin1, cos2, sin2], dim=-1)  # (B, L, Kc, 4)

    def forward(  # type: ignore[override]
        self,
        X: torch.Tensor,
        residue_mask: torch.Tensor,
        R_idx: torch.Tensor,
        chain_labels: torch.Tensor,
        Y: torch.Tensor,
        Y_m: torch.Tensor,
        Y_t: torch.Tensor,
        structure_noise: float = 0.0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute graph features for protein-ligand structures.

        Args:
            X: Backbone coordinates, shape ``(B, L, 4, 3)``.
            residue_mask: Residue validity, shape ``(B, L)``.
            R_idx: Residue indices, shape ``(B, L)``.
            chain_labels: Chain identifiers, shape ``(B, L)``.
            Y: Non-protein atom coordinates, shape ``(B, N, 3)``.
            Y_m: Non-protein atom mask, shape ``(B, N)``.
            Y_t: Non-protein atom types, shape ``(B, N)``.
            structure_noise: Gaussian noise std dev.

        Returns:
            Dict extending parent with ``E_protein_to_ligand``,
            ``ligand_subgraph_nodes``, ``ligand_subgraph_edges``,
            ``ligand_subgraph_Y_m``, ``Y_idx``.
        """
        # Get parent protein backbone features
        result = super().forward(X, residue_mask, R_idx, chain_labels, structure_noise)

        # Add noise to coordinates for feature computation
        if structure_noise > 0.0 and self.training:
            X_noisy = X + torch.randn_like(X) * structure_noise
        else:
            X_noisy = X

        B, L = X.shape[:2]
        N_atoms = Y.shape[1]
        if N_atoms == 0:
            empty_p2l = X.new_zeros(B, L, 0, self.hidden_dim)
            empty_nodes = X.new_zeros(B, 0, self.hidden_dim)
            empty_edges = X.new_zeros(B, 0, 0, self.hidden_dim)
            empty_mask_context = Y_m.new_zeros(B, L, 0)
            empty_mask_edges = Y_m.new_zeros(B, 0, 0)
            empty_idx_context = torch.zeros(B, L, 0, dtype=torch.long, device=X.device)
            empty_idx_edges = torch.zeros(B, 0, 0, dtype=torch.long, device=X.device)
            result.update(
                {
                    "E_protein_to_ligand": empty_p2l,
                    "ligand_subgraph_nodes": empty_nodes,
                    "ligand_subgraph_edges": empty_edges,
                    "ligand_subgraph_Y_m": empty_mask_context,
                    "ligand_subgraph_Y_m_edges": empty_mask_edges,
                    "ligand_subgraph_E_idx": empty_idx_edges,
                    "Y_idx": empty_idx_context,
                }
            )
            return result

        # Compute virtual CB for distance calculations
        CB = compute_virtual_cb(X_noisy).squeeze(-2)  # (B, L, 3)

        # --- Find nearest context atoms per residue ---
        # Distance from each residue CB to each ligand atom: (B, L, N_atoms)
        cb_to_y = (CB.unsqueeze(2) - Y.unsqueeze(1)).norm(dim=-1)
        # Mask invalid atoms with large distance
        cb_to_y = cb_to_y + (~Y_m.bool()).unsqueeze(1).float() * 1e6

        Kc = min(self.num_context_atoms, N_atoms)
        # Select top-Kc nearest: (B, L, Kc)
        _, Y_idx = cb_to_y.topk(Kc, dim=-1, largest=False)

        # Gather context atom coords and types: (B, L, Kc, ...)
        Y_context = torch.gather(
            Y.unsqueeze(1).expand(B, L, N_atoms, 3),
            2,
            Y_idx.unsqueeze(-1).expand(B, L, Kc, 3),
        )
        Y_m_context = torch.gather(
            Y_m.unsqueeze(1).expand(B, L, N_atoms),
            2,
            Y_idx,
        )
        Y_t_context = torch.gather(
            Y_t.unsqueeze(1).expand(B, L, N_atoms),
            2,
            Y_idx,
        )

        # --- Protein-to-ligand edge features ---
        # RBF from 5 backbone atoms to each context atom: (B, L, Kc, 80)
        atoms_5 = torch.cat([X_noisy, compute_virtual_cb(X_noisy)], dim=-2)  # (B, L, 5, 3)
        p2l_rbf_list = []
        for a in range(5):
            src = atoms_5[:, :, a, :]  # (B, L, 3)
            dist = (src.unsqueeze(2) - Y_context).norm(dim=-1)  # (B, L, Kc)
            p2l_rbf_list.append(rbf_encode(dist, num_rbf=self.num_rbf))
        p2l_rbf = torch.cat(p2l_rbf_list, dim=-1)  # (B, L, Kc, 80)

        # Atom type embedding: (B, L, Kc, 146) → (B, L, Kc, 64)
        atom_type_raw = self._encode_atom_type(Y_t_context)
        atom_type_embed = self.embed_atom_type_features(atom_type_raw)

        # Angle features: (B, L, Kc, 4)
        angles = self._compute_angle_features(X_noisy, Y_context)

        # Concatenate and project: (B, L, Kc, 148) → (B, L, Kc, 128)
        p2l_features = torch.cat([p2l_rbf, atom_type_embed, angles], dim=-1)
        E_protein_to_ligand = self.node_norm(self.node_embedding(p2l_features))
        E_protein_to_ligand = E_protein_to_ligand * Y_m_context.unsqueeze(-1)

        # --- Ligand subgraph nodes ---
        # Global atom type features for all ligand atoms: (B, N, 146) → (B, N, 128)
        atom_type_all = self._encode_atom_type(Y_t)
        ligand_nodes = self.ligand_subgraph_node_norm(
            self.ligand_subgraph_node_embedding(atom_type_all)
        )
        ligand_nodes = ligand_nodes * Y_m.unsqueeze(-1)

        # --- Ligand subgraph edges ---
        # Inter-atom distances for k-NN within ligand: (B, N, N)
        y_diff = Y.unsqueeze(2) - Y.unsqueeze(1)
        y_dist = y_diff.norm(dim=-1)
        y_dist = y_dist + (~Y_m.bool()).unsqueeze(1).float() * 1e6

        Ke = min(self.num_context_atoms, N_atoms)
        _, Y_edges_idx = y_dist.topk(Ke, dim=-1, largest=False)

        # RBF for intraligand edges: (B, N, Ke, 16) → (B, N, Ke, 128)
        y_neighbor_coords = torch.gather(
            Y.unsqueeze(2).expand(B, N_atoms, Ke, 3),
            1,
            Y_edges_idx.unsqueeze(-1).expand(B, N_atoms, Ke, 3),
        )
        y_edge_dist = (Y.unsqueeze(2) - y_neighbor_coords).norm(dim=-1)
        y_edge_rbf = rbf_encode(y_edge_dist, num_rbf=self.num_rbf)
        ligand_edges = self.ligand_subgraph_edge_norm(
            self.ligand_subgraph_edge_embedding(y_edge_rbf)
        )
        Y_m_edges = torch.gather(
            Y_m.unsqueeze(2).expand(B, N_atoms, Ke),
            1,
            Y_edges_idx,
        )
        ligand_edges = ligand_edges * Y_m_edges.unsqueeze(-1)

        result.update(
            {
                "E_protein_to_ligand": E_protein_to_ligand,
                "ligand_subgraph_nodes": ligand_nodes,
                "ligand_subgraph_edges": ligand_edges,
                "ligand_subgraph_Y_m": Y_m_context,
                "ligand_subgraph_Y_m_edges": Y_m_edges,
                "ligand_subgraph_E_idx": Y_edges_idx,
                "Y_idx": Y_idx,
            }
        )
        return result
