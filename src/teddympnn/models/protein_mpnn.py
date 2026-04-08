"""ProteinMPNN model implementation.

Reimplements the ProteinMPNN architecture with exact Foundry-compatible module
hierarchy and attribute names for bidirectional checkpoint transfer.

Architecture: 3 encoder layers + 3 decoder layers, 128-dim hidden state,
k=48 neighbors, 21 amino acid vocabulary. ~1.66M parameters.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from teddympnn.models.layers.graph_embeddings import ProteinFeatures
from teddympnn.models.layers.message_passing import (
    DecLayer,
    EncLayer,
    cat_neighbors_nodes,
    gather_nodes,
)
from teddympnn.models.tokens import VOCAB_SIZE


class ProteinMPNN(nn.Module):
    """ProteinMPNN message passing neural network.

    Encodes protein backbone structure through a k-NN graph with encoder
    message passing layers, then autoregressively decodes amino acid sequences
    through decoder layers with causal masking.

    Args:
        hidden_dim: Hidden state dimensionality (default 128).
        num_encoder_layers: Number of encoder message passing layers (default 3).
        num_decoder_layers: Number of decoder message passing layers (default 3).
        num_neighbors: k-NN neighbor count (default 48).
        vocab_size: Amino acid vocabulary size (default 21).
        dropout: Dropout probability (default 0.1).
        augment_eps: Backbone coordinate noise (default 0.0).
        num_positional_embeddings: Positional encoding dim (default 16).
        num_rbf: RBF kernel count (default 16).
        max_relative_feature: Max relative position (default 32).
        use_gradient_checkpointing: Enable gradient checkpointing (default False).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        num_neighbors: int = 48,
        vocab_size: int = VOCAB_SIZE,
        dropout: float = 0.1,
        augment_eps: float = 0.0,
        num_positional_embeddings: int = 16,
        num_rbf: int = 16,
        max_relative_feature: int = 32,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_neighbors = num_neighbors
        self.vocab_size = vocab_size
        self.augment_eps = augment_eps
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Graph featurization
        self.graph_featurization_module = ProteinFeatures(
            num_positional_embeddings=num_positional_embeddings,
            num_rbf=num_rbf,
            top_k=num_neighbors,
            hidden_dim=hidden_dim,
            max_relative_feature=max_relative_feature,
            dropout=dropout,
        )

        # Edge projection
        self.W_e = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # Sequence embedding
        self.W_s = nn.Embedding(vocab_size, hidden_dim)

        # Encoder: num_in = 3H (source_node + neighbor_node + edge)
        enc_num_in = 3 * hidden_dim
        self.encoder_layers = nn.ModuleList(
            [EncLayer(hidden_dim, enc_num_in, dropout=dropout) for _ in range(num_encoder_layers)]
        )

        # Decoder: num_in = 4H (source_node + neighbor_node + edge + sequence)
        dec_num_in = 4 * hidden_dim
        self.decoder_layers = nn.ModuleList(
            [DecLayer(hidden_dim, dec_num_in, dropout=dropout) for _ in range(num_decoder_layers)]
        )

        # Output projection
        self.W_out = nn.Linear(hidden_dim, vocab_size, bias=True)

    def _compute_graph_features(
        self,
        input_features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute graph features from input batch.

        Override this in LigandMPNN to pass additional arguments.
        """
        return self.graph_featurization_module(  # type: ignore[no-any-return]
            X=input_features["X"],
            residue_mask=input_features["residue_mask"],
            R_idx=input_features["R_idx"],
            chain_labels=input_features["chain_labels"],
            structure_noise=self.augment_eps if self.training else 0.0,
        )

    def encode(
        self,
        input_features: dict[str, torch.Tensor],
        graph_features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Run encoder message passing.

        Args:
            input_features: Batch dict with ``residue_mask``.
            graph_features: Dict with ``E_idx`` and ``E``.

        Returns:
            Dict with ``h_V`` (B, L, H) and ``h_E`` (B, L, K, H).
        """
        E_idx = graph_features["E_idx"]
        B, L = input_features["residue_mask"].shape
        K = E_idx.shape[2]

        # Initialize node states as zeros
        h_V = torch.zeros(B, L, self.hidden_dim, device=E_idx.device, dtype=E_idx.dtype).float()
        h_E = self.W_e(graph_features["E"])

        # Compute edge mask from residue mask
        mask_V = input_features["residue_mask"].float()
        # Gather mask at neighbor indices
        mask_attend = torch.gather(
            mask_V.unsqueeze(-1).expand(B, L, K),
            dim=1,
            index=E_idx,
        )
        # AND with self mask
        mask_attend = mask_attend * mask_V.unsqueeze(-1)

        # Message passing
        for layer in self.encoder_layers:
            if self.use_gradient_checkpointing and self.training:
                h_V, h_E = gradient_checkpoint(
                    layer,
                    h_V,
                    h_E,
                    E_idx,
                    mask_V,
                    mask_attend,
                    use_reentrant=False,
                )
            else:
                h_V, h_E = layer(h_V, h_E, E_idx, mask_V, mask_attend)

        return {"h_V": h_V, "h_E": h_E}

    def _setup_causality_masks(
        self,
        input_features: dict[str, torch.Tensor],
        graph_features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Generate random decoding order and causal/anti-causal masks.

        Args:
            input_features: Batch dict with ``designed_residue_mask``, ``residue_mask``.
            graph_features: Dict with ``E_idx``.

        Returns:
            Dict with ``decoding_order``, ``causal_mask``, ``anti_causal_mask``.
        """
        E_idx = graph_features["E_idx"]
        designed_mask = input_features["designed_residue_mask"]
        residue_mask = input_features["residue_mask"]
        B, L = residue_mask.shape
        K = E_idx.shape[2]
        device = E_idx.device

        # Random order for designed positions, fixed (non-designed) go first
        # Assign random values to designed positions, 0 to fixed
        noise = torch.rand(B, L, device=device) * designed_mask.float()
        # Fixed positions get -1 so they sort first (decoded "before" designed)
        noise = noise - (1.0 - designed_mask.float())
        # Sort to get decoding order (ascending)
        _, decoding_order = noise.sort(dim=-1)

        # Build permutation mask: position_order[b, decoding_order[b, i]] = i
        position_order = torch.zeros_like(decoding_order)
        arange_L = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        position_order.scatter_(1, decoding_order, arange_L)

        # Causal mask: neighbor j is visible to position i if j is decoded before i
        # position_order[b, E_idx[b, i, k]] < position_order[b, i] → visible
        order_i = torch.gather(position_order, 1, arange_L)
        order_i = order_i.unsqueeze(-1).expand(B, L, K)  # (B, L, K)
        order_j = torch.gather(
            position_order.unsqueeze(-1).expand(B, L, K),
            dim=1,
            index=E_idx,
        )

        # Causal: neighbor was decoded before current position
        causal_mask = (order_j < order_i).float().unsqueeze(-1)  # (B, L, K, 1)
        anti_causal_mask = 1.0 - causal_mask

        return {
            "decoding_order": decoding_order,
            "causal_mask": causal_mask,
            "anti_causal_mask": anti_causal_mask,
        }

    def decode_teacher_forcing(
        self,
        input_features: dict[str, torch.Tensor],
        graph_features: dict[str, torch.Tensor],
        encoder_output: dict[str, torch.Tensor],
        causality_masks: dict[str, torch.Tensor],
        temperature: float = 1.0,
        bias: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode with teacher forcing (training mode).

        Args:
            input_features: Batch dict with ``S``, ``residue_mask``.
            graph_features: Dict with ``E_idx``.
            encoder_output: Dict with ``h_V``, ``h_E``.
            causality_masks: Dict with ``causal_mask``, ``anti_causal_mask``.
            temperature: Softmax temperature.
            bias: Optional logit bias, shape ``(vocab_size,)``.

        Returns:
            Dict with ``log_probs`` (B, L, vocab_size).
        """
        E_idx = graph_features["E_idx"]
        h_V_enc = encoder_output["h_V"]
        h_E = encoder_output["h_E"]
        S = input_features["S"]
        mask_V = input_features["residue_mask"].float()
        causal_mask = causality_masks["causal_mask"]
        anti_causal_mask = causality_masks["anti_causal_mask"]

        B, L = S.shape

        # Embed ground truth sequence
        h_S = self.W_s(S)  # (B, L, H)

        # Pre-compute encoder anti-causal features:
        # [h_E, zeros_seq, h_V_enc_neighbors] = 3H
        h_EX_encoder = cat_neighbors_nodes(
            torch.zeros_like(h_V_enc), h_E, E_idx
        )  # [h_E, zeros] = 2H
        h_EXV_encoder = cat_neighbors_nodes(
            h_V_enc, h_EX_encoder, E_idx
        )  # [h_EX_encoder, h_V_enc_neighbors] = 3H
        h_EXV_encoder_anti_causal = h_EXV_encoder * anti_causal_mask

        # Compute edge mask
        mask_attend = gather_nodes(mask_V.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask_V.unsqueeze(-1) * mask_attend

        # Pre-compute decoder edge+sequence features: [h_E, h_S_neighbors] = 2H
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)  # (B, L, K, 2H)

        # Initialize decoder from encoder output (Foundry convention)
        h_V_dec = h_V_enc.clone()

        for layer in self.decoder_layers:
            # Decoder features: [h_ES, h_V_dec_neighbors] = 3H
            h_ESV_decoder = cat_neighbors_nodes(h_V_dec, h_ES, E_idx)  # (B, L, K, 3H)

            # Combine with causal mask: 3H
            h_ESV = causal_mask * h_ESV_decoder + h_EXV_encoder_anti_causal

            # DecLayer internally prepends source h_V_dec → 4H
            if self.use_gradient_checkpointing and self.training:
                h_V_dec = gradient_checkpoint(
                    layer,
                    h_V_dec,
                    h_ESV,
                    mask_V,
                    mask_attend,
                    use_reentrant=False,
                )
            else:
                h_V_dec = layer(h_V_dec, h_ESV, mask_V, mask_attend)

        # Output logits
        logits = self.W_out(h_V_dec)  # (B, L, vocab_size)
        if bias is not None:
            logits = logits + bias
        log_probs = F.log_softmax(logits / temperature, dim=-1)

        return {"log_probs": log_probs}

    def decode_autoregressive(
        self,
        input_features: dict[str, torch.Tensor],
        graph_features: dict[str, torch.Tensor],
        encoder_output: dict[str, torch.Tensor],
        causality_masks: dict[str, torch.Tensor],
        temperature: float = 1.0,
        bias: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Decode autoregressively (inference mode).

        Args:
            input_features: Batch dict with ``S``, ``residue_mask``,
                ``designed_residue_mask``.
            graph_features: Dict with ``E_idx``.
            encoder_output: Dict with ``h_V``, ``h_E``.
            causality_masks: Dict with ``decoding_order``, ``causal_mask``,
                ``anti_causal_mask``.
            temperature: Sampling temperature.
            bias: Optional logit bias.

        Returns:
            Dict with ``S_sample`` (B, L), ``log_probs`` (B, L, vocab_size).
        """
        E_idx = graph_features["E_idx"]
        h_V_enc = encoder_output["h_V"]
        h_E = encoder_output["h_E"]
        S = input_features["S"].clone()
        mask_V = input_features["residue_mask"].float()
        designed_mask = input_features["designed_residue_mask"]
        decoding_order = causality_masks["decoding_order"]

        B, L = S.shape
        K = E_idx.shape[2]

        # Pre-compute encoder anti-causal features: [h_E, zeros, h_V_enc] = 3H
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_V_enc), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V_enc, h_EX_encoder, E_idx)

        # Edge mask
        mask_attend = gather_nodes(mask_V.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = mask_V.unsqueeze(-1) * mask_attend

        # Track decoded sequence and log probs
        S_sample = S.clone()
        all_log_probs = torch.zeros(B, L, self.vocab_size, device=S.device)

        # Build position order for causal masking
        position_order = torch.zeros_like(decoding_order)
        position_order.scatter_(
            1,
            decoding_order,
            torch.arange(L, device=S.device).unsqueeze(0).expand(B, L),
        )

        for step in range(L):
            pos = decoding_order[:, step]

            h_S = self.W_s(S_sample)

            # Build per-step causal mask
            order_i = position_order.unsqueeze(-1).expand(B, L, K)
            order_j = torch.gather(
                position_order.unsqueeze(-1).expand(B, L, K),
                dim=1,
                index=E_idx,
            )
            step_causal = (order_j < order_i).float().unsqueeze(-1)
            step_anti_causal = 1.0 - step_causal

            h_EXV_enc_anti = h_EXV_encoder * step_anti_causal
            h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

            h_V_dec_step = h_V_enc.clone()
            for layer in self.decoder_layers:
                h_ESV_decoder = cat_neighbors_nodes(h_V_dec_step, h_ES, E_idx)
                h_ESV = step_causal * h_ESV_decoder + h_EXV_enc_anti
                h_V_dec_step = layer(h_V_dec_step, h_ESV, mask_V, mask_attend)

            # Get logits at current position
            logits = self.W_out(h_V_dec_step)
            if bias is not None:
                logits = logits + bias
            log_probs = F.log_softmax(logits / temperature, dim=-1)

            # Sample at the current decoding position
            for b in range(B):
                p = int(pos[b].item())
                if designed_mask[b, p]:
                    probs = torch.exp(log_probs[b, p])
                    S_sample[b, p] = torch.multinomial(probs, 1).squeeze(-1)
                all_log_probs[b, p] = log_probs[b, p]

        return {"S_sample": S_sample, "log_probs": all_log_probs}

    def forward(
        self,
        input_features: dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Forward pass: encode structure, decode sequence with teacher forcing.

        Args:
            input_features: Batch dict with keys ``X``, ``S``, ``R_idx``,
                ``chain_labels``, ``residue_mask``, ``designed_residue_mask``.

        Returns:
            Dict with ``log_probs`` (B, L, vocab_size).
        """
        graph_features = self._compute_graph_features(input_features)
        encoder_output = self.encode(input_features, graph_features)
        causality_masks = self._setup_causality_masks(input_features, graph_features)
        return self.decode_teacher_forcing(
            input_features,
            graph_features,
            encoder_output,
            causality_masks,
            **kwargs,
        )

    @torch.no_grad()
    def score(
        self,
        input_features: dict[str, torch.Tensor],
        score_mask: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Score a structure-sequence pair (per-residue log-probabilities).

        Uses teacher forcing. Returns log p(s_i | X, s_{<i}) at the ground
        truth token for each position.

        Args:
            input_features: Batch dict.
            score_mask: Optional mask for which residues to score. If None,
                scores all residues marked in ``designed_residue_mask``.
            temperature: Softmax temperature.

        Returns:
            Per-residue log-probabilities, shape ``(B, L)``.
        """
        self.eval()
        graph_features = self._compute_graph_features(input_features)
        encoder_output = self.encode(input_features, graph_features)
        causality_masks = self._setup_causality_masks(input_features, graph_features)
        decode_output = self.decode_teacher_forcing(
            input_features,
            graph_features,
            encoder_output,
            causality_masks,
            temperature=temperature,
        )
        log_probs = decode_output["log_probs"]  # (B, L, V)
        S = input_features["S"]  # (B, L)

        # Gather log prob at ground truth token: (B, L)
        per_residue = torch.gather(log_probs, 2, S.unsqueeze(-1)).squeeze(-1)

        if score_mask is not None:
            per_residue = per_residue * score_mask.float()

        return per_residue

    @torch.no_grad()
    def sample(
        self,
        input_features: dict[str, torch.Tensor],
        temperature: float = 1.0,
        bias: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample sequences autoregressively.

        Args:
            input_features: Batch dict.
            temperature: Sampling temperature.
            bias: Optional amino acid logit bias.

        Returns:
            Dict with ``S_sample`` and ``log_probs``.
        """
        self.eval()
        graph_features = self._compute_graph_features(input_features)
        encoder_output = self.encode(input_features, graph_features)
        causality_masks = self._setup_causality_masks(input_features, graph_features)
        return self.decode_autoregressive(
            input_features,
            graph_features,
            encoder_output,
            causality_masks,
            temperature=temperature,
            bias=bias,
        )
