"""LigandMPNN model implementation.

Extends ProteinMPNN with a protein-ligand context encoder that processes
three graphs: protein backbone, protein-to-ligand, and intraligand.

Architecture: ProteinMPNN base + 2 ligand context encoder layers + 2
protein-ligand context encoder layers. ~2.62M parameters.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from teddympnn.models.layers.graph_embeddings import ProteinFeaturesLigand
from teddympnn.models.layers.message_passing import DecLayer
from teddympnn.models.protein_mpnn import ProteinMPNN
from teddympnn.models.tokens import VOCAB_SIZE


class LigandMPNN(ProteinMPNN):
    """LigandMPNN with protein-ligand context encoder.

    Extends ProteinMPNN by replacing the graph featurization module with
    ``ProteinFeaturesLigand`` and adding context encoder layers that process
    ligand atom features before the standard decoder.

    Args:
        hidden_dim: Hidden state dimensionality (default 128).
        num_encoder_layers: Backbone encoder layers (default 3).
        num_decoder_layers: Decoder layers (default 3).
        num_neighbors: Protein backbone k-NN count (default 32, reduced from 48).
        vocab_size: Amino acid vocabulary size (default 21).
        dropout: Dropout probability (default 0.1).
        augment_eps: Backbone coordinate noise (default 0.0).
        num_positional_embeddings: Positional encoding dim (default 16).
        num_rbf: RBF kernel count (default 16).
        max_relative_feature: Max relative position (default 32).
        num_context_encoder_layers: Protein-ligand context layers (default 2).
        num_ligand_encoder_layers: Intraligand context layers (default 2).
        use_gradient_checkpointing: Enable gradient checkpointing (default False).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        num_neighbors: int = 32,
        vocab_size: int = VOCAB_SIZE,
        dropout: float = 0.1,
        augment_eps: float = 0.0,
        num_positional_embeddings: int = 16,
        num_rbf: int = 16,
        max_relative_feature: int = 32,
        num_context_atoms: int = 25,
        num_context_encoder_layers: int = 2,
        num_ligand_encoder_layers: int = 2,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_neighbors=num_neighbors,
            vocab_size=vocab_size,
            dropout=dropout,
            augment_eps=augment_eps,
            num_positional_embeddings=num_positional_embeddings,
            num_rbf=num_rbf,
            max_relative_feature=max_relative_feature,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        # Replace graph featurization with ligand-aware version
        self.graph_featurization_module = ProteinFeaturesLigand(
            num_positional_embeddings=num_positional_embeddings,
            num_rbf=num_rbf,
            top_k=num_neighbors,
            hidden_dim=hidden_dim,
            max_relative_feature=max_relative_feature,
            dropout=dropout,
            num_context_atoms=num_context_atoms,
        )

        # Ligand context embedding projections
        self.W_protein_to_ligand_edges_embed = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_protein_encoding_embed = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_ligand_nodes_embed = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_ligand_edges_embed = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Final context projection (no bias, as in Foundry)
        self.W_final_context_embed = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.final_context_norm = nn.LayerNorm(hidden_dim)

        # Protein-ligand context encoder: num_in = 3H (source + edge + ligand_node)
        context_num_in = 3 * hidden_dim
        self.protein_ligand_context_encoder_layers = nn.ModuleList(
            [
                DecLayer(hidden_dim, context_num_in, dropout=dropout)
                for _ in range(num_context_encoder_layers)
            ]
        )

        # Ligand subgraph context encoder: num_in = 2H (source + edge)
        ligand_num_in = 2 * hidden_dim
        self.ligand_context_encoder_layers = nn.ModuleList(
            [
                DecLayer(hidden_dim, ligand_num_in, dropout=dropout)
                for _ in range(num_ligand_encoder_layers)
            ]
        )

    def _compute_graph_features(
        self,
        input_features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute graph features including ligand context."""
        return self.graph_featurization_module(  # type: ignore[no-any-return]
            X=input_features["X"],
            residue_mask=input_features["residue_mask"],
            R_idx=input_features["R_idx"],
            chain_labels=input_features["chain_labels"],
            Y=input_features["Y"],
            Y_m=input_features["Y_m"],
            Y_t=input_features["Y_t"],
            structure_noise=self.augment_eps if self.training else 0.0,
        )

    def encode(
        self,
        input_features: dict[str, torch.Tensor],
        graph_features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Run encoder with ligand context integration.

        First runs the parent protein backbone encoder, then processes the
        ligand context through intraligand and protein-ligand context
        encoder layers, and adds the result to protein node embeddings.

        Args:
            input_features: Batch dict with ``residue_mask``.
            graph_features: Dict with backbone and ligand graph features.

        Returns:
            Dict with ``h_V`` (B, L, H) and ``h_E`` (B, L, K, H).
        """
        # Run parent protein backbone encoder
        encoder_output = super().encode(input_features, graph_features)
        h_V = encoder_output["h_V"]
        h_E = encoder_output["h_E"]

        mask_V = input_features["residue_mask"].float()

        # Extract ligand graph features
        E_protein_to_ligand = graph_features["E_protein_to_ligand"]
        ligand_nodes = graph_features["ligand_subgraph_nodes"]
        ligand_edges = graph_features["ligand_subgraph_edges"]
        Y_m_context = graph_features["ligand_subgraph_Y_m"]
        Y_m_edges = graph_features["ligand_subgraph_Y_m_edges"]
        valid_context = Y_m_context.bool().any(dim=-1).float().unsqueeze(-1)

        # Embed ligand features
        h_E_p2l = self.W_protein_to_ligand_edges_embed(E_protein_to_ligand)
        h_V_context = self.W_protein_encoding_embed(h_V)
        h_nodes = self.W_ligand_nodes_embed(ligand_nodes)
        h_edges = self.W_ligand_edges_embed(ligand_edges)

        # Masks for ligand atoms
        Y_m_float = input_features["Y_m"].float()
        Y_idx = graph_features["Y_idx"]

        # Interleaved context encoding
        for i in range(len(self.ligand_context_encoder_layers)):
            # --- Intraligand message passing ---
            # Pass only edge features (H); DecLayer prepends source (H) → 2H
            # NOTE: Foundry does NOT concatenate destination node features here
            # (documented as a bug in the original LigandMPNN, but preserved
            # for weight compatibility).
            if self.use_gradient_checkpointing and self.training:
                h_nodes = gradient_checkpoint(
                    self.ligand_context_encoder_layers[i],
                    h_nodes,
                    h_edges,
                    Y_m_float,
                    Y_m_edges,
                    use_reentrant=False,
                )
            else:
                h_nodes = self.ligand_context_encoder_layers[i](
                    h_nodes,
                    h_edges,
                    Y_m_float,
                    Y_m_edges,
                )

            # --- Protein-to-ligand context message passing ---
            # Gather updated ligand nodes at context atom indices: (B, L, Kc, H)
            h_context_nodes = torch.gather(
                h_nodes.unsqueeze(1).expand(-1, h_V.shape[1], -1, -1),
                dim=2,
                index=Y_idx.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim),
            )

            # Neighbor features: cat(p2l_edge, context_nodes) = 2H
            # DecLayer prepends source h_V_context → 3H total
            h_E_context = torch.cat([h_E_p2l, h_context_nodes], dim=-1)  # (B, L, Kc, 2H)

            if self.use_gradient_checkpointing and self.training:
                h_V_context = gradient_checkpoint(
                    self.protein_ligand_context_encoder_layers[i],
                    h_V_context,
                    h_E_context,
                    mask_V,
                    Y_m_context,
                    use_reentrant=False,
                )
            else:
                h_V_context = self.protein_ligand_context_encoder_layers[i](
                    h_V_context,
                    h_E_context,
                    mask_V,
                    Y_m_context,
                )

        # Add context to protein node embeddings
        context_contribution = self.final_context_norm(
            self.graph_featurization_module.dropout(self.W_final_context_embed(h_V_context))
        )
        context_contribution = context_contribution * valid_context
        h_V = h_V + context_contribution

        return {"h_V": h_V, "h_E": h_E}

    def forward(
        self,
        input_features: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with ligand context.

        Requires additional keys in ``input_features``: ``Y``, ``Y_m``, ``Y_t``.
        """
        return super().forward(input_features, **kwargs)
