"""Tests for the Noam learning rate scheduler."""

from __future__ import annotations

import torch

from teddympnn.training.scheduler import NoamScheduler


def _make_optimizer() -> torch.optim.Adam:
    """Create a dummy optimizer for testing."""
    param = torch.nn.Parameter(torch.randn(10))
    return torch.optim.Adam([param], lr=1.0)


class TestNoamScheduler:
    def test_lr_zero_at_step_zero(self) -> None:
        """LR should be 0 at step 0 to prevent division by zero."""
        optimizer = _make_optimizer()
        NoamScheduler(optimizer, d_model=128, warmup_steps=4000, factor=2.0)
        # After initialization (before any step), LR should be 0
        # LambdaLR initializes by calling the lambda at last_epoch+1=0
        assert optimizer.param_groups[0]["lr"] == 0.0

    def test_lr_increases_during_warmup(self) -> None:
        """LR should increase monotonically during warmup."""
        optimizer = _make_optimizer()
        scheduler = NoamScheduler(optimizer, d_model=128, warmup_steps=4000, factor=2.0)

        prev_lr = 0.0
        for _ in range(100):
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            assert lr > prev_lr, f"LR should increase during warmup: {lr} <= {prev_lr}"
            prev_lr = lr

    def test_lr_peaks_near_warmup_steps(self) -> None:
        """LR should peak approximately at warmup_steps."""
        optimizer = _make_optimizer()
        warmup = 4000
        scheduler = NoamScheduler(optimizer, d_model=128, warmup_steps=warmup, factor=2.0)

        lrs = []
        for _ in range(8000):
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        peak_step = lrs.index(max(lrs)) + 1  # +1 because step() increments
        # Peak should be near warmup_steps
        assert abs(peak_step - warmup) <= 10

    def test_lr_decays_after_warmup(self) -> None:
        """LR should decay as step^(-0.5) after warmup."""
        optimizer = _make_optimizer()
        warmup = 100
        scheduler = NoamScheduler(optimizer, d_model=128, warmup_steps=warmup, factor=2.0)

        # Advance past warmup
        for _ in range(warmup + 50):
            scheduler.step()

        prev_lr = optimizer.param_groups[0]["lr"]
        for _ in range(100):
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            assert lr < prev_lr, "LR should decay after warmup"
            prev_lr = lr

    def test_state_dict_roundtrip(self) -> None:
        """Scheduler state can be saved and restored, producing same LR on next step."""
        optimizer = _make_optimizer()
        scheduler = NoamScheduler(optimizer, d_model=128, warmup_steps=4000, factor=2.0)

        for _ in range(50):
            scheduler.step()
        state = scheduler.state_dict()

        # Step once more to get the reference LR at step 51
        scheduler.step()
        lr_at_51 = optimizer.param_groups[0]["lr"]

        # Create fresh scheduler, load state (at step 50), then step to 51
        optimizer2 = _make_optimizer()
        scheduler2 = NoamScheduler(optimizer2, d_model=128, warmup_steps=4000, factor=2.0)
        scheduler2.load_state_dict(state)
        scheduler2.step()

        assert optimizer2.param_groups[0]["lr"] == lr_at_51
