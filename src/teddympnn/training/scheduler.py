"""Noam learning rate scheduler for transformer-style warmup + decay."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.optim.lr_scheduler import LambdaLR

if TYPE_CHECKING:
    from torch.optim import Optimizer


class NoamScheduler(LambdaLR):
    """Noam learning rate schedule: linear warmup then inverse-sqrt decay.

    LR at step *t*:
        ``factor * d_model^(-0.5) * min(t^(-0.5), t * warmup_steps^(-1.5))``

    Step 0 returns LR = 0 to prevent division by zero.

    Args:
        optimizer: Wrapped optimizer.
        d_model: Model hidden dimension (default 128).
        warmup_steps: Linear warmup duration (default 4000).
        factor: Scaling factor (default 2).
        last_epoch: Index of last epoch (default -1).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int = 128,
        warmup_steps: int = 4000,
        factor: float = 2.0,
        last_epoch: int = -1,
    ) -> None:
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor

        def lr_lambda(step: int) -> float:
            if step == 0:
                return 0.0
            return float(
                self.factor
                * (self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5)))
            )

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)
