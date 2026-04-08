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
    ) -> torch.Tensor:
        """Compute masked label-smoothed NLL loss.

        Args:
            log_probs: Predicted log-probabilities, shape ``(B, L, V)``.
            targets: Ground-truth token indices, shape ``(B, L)``.
            mask: Loss mask (1 = designed position), shape ``(B, L)``.

        Returns:
            Scalar loss (mean over designed positions).
        """
        # One-hot encode targets: (B, L, V)
        one_hot = torch.zeros_like(log_probs).scatter_(2, targets.unsqueeze(-1), 1.0)

        # Apply label smoothing
        eps = self.label_smoothing
        smoothed = (1.0 - eps) * one_hot + eps / self.vocab_size

        # Per-residue NLL: -(smoothed * log_probs).sum(dim=-1) → (B, L)
        per_residue_nll = -(smoothed * log_probs).sum(dim=-1)

        # Mask and reduce
        mask_float = mask.float()
        numerator = (per_residue_nll * mask_float).sum()
        denominator = mask_float.sum()

        # DDP reduction: sum numerator and denominator across workers
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(numerator)
            torch.distributed.all_reduce(denominator)

        return numerator / denominator.clamp(min=1.0)
