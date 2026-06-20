"""Label-smoothed negative log-likelihood loss for sequence design."""

from __future__ import annotations

import torch
from torch import nn


class LabelSmoothedNLLLoss(nn.Module):
    """Cross-entropy loss with label smoothing for amino acid prediction.

    Computes the negative log-likelihood against smoothed one-hot targets,
    masked to only count designed positions. In DDP mode, numerator and
    denominator are reduced across workers before division so the effective
    loss equals the mean over all designed positions globally.

    Args:
        label_smoothing: Smoothing factor (default 0.1).
        vocab_size: Number of amino acid classes (default 21).
    """

    def __init__(
        self,
        label_smoothing: float = 0.1,
        vocab_size: int = 21,
    ) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.vocab_size = vocab_size

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute masked, optionally per-residue-weighted, label-smoothed NLL.

        Args:
            log_probs: Predicted log-probabilities, shape ``(B, L, V)``.
            targets: Ground-truth token indices, shape ``(B, L)``.
            mask: Loss mask (1 = designed position), shape ``(B, L)``.
            weights: Optional per-residue weights, shape ``(B, L)``. When given,
                the loss becomes the weighted mean over designed positions;
                ``None`` (or all-equal weights) reproduces the unweighted mean.

        Returns:
            Scalar loss (weighted mean over designed positions).
        """
        # One-hot encode targets: (B, L, V)
        one_hot = torch.zeros_like(log_probs).scatter_(2, targets.unsqueeze(-1), 1.0)

        # Apply label smoothing
        eps = self.label_smoothing
        smoothed = (1.0 - eps) * one_hot + eps / self.vocab_size

        # Per-residue NLL: (B, L)
        per_residue_nll = -(smoothed * log_probs).sum(dim=-1)

        # Effective per-residue weight = mask, optionally scaled.
        weight = mask.float()
        if weights is not None:
            weight = weight * weights.to(weight.dtype)

        numerator = (per_residue_nll * weight).sum()
        denominator = weight.sum()

        # DDP reduction: sum numerator and denominator across workers
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(numerator)
            torch.distributed.all_reduce(denominator)

        return numerator / denominator.clamp(min=1.0)
