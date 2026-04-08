"""Tests for the label-smoothed NLL loss."""

from __future__ import annotations

import math

import torch

from teddympnn.training.loss import LabelSmoothedNLLLoss


class TestLabelSmoothedNLLLoss:
    def test_perfect_predictions_lower_than_uniform(self) -> None:
        """Near-perfect predictions give lower loss than uniform."""
        loss_fn = LabelSmoothedNLLLoss(label_smoothing=0.1, vocab_size=21)
        B, L, V = 2, 10, 21
        targets = torch.randint(0, V, (B, L))

        # Near-perfect: high logit at correct token
        logits_perfect = torch.full((B, L, V), -10.0)
        logits_perfect.scatter_(2, targets.unsqueeze(-1), 10.0)
        log_probs_perfect = torch.log_softmax(logits_perfect, dim=-1)

        # Uniform
        log_probs_uniform = torch.full((B, L, V), math.log(1.0 / V))

        mask = torch.ones(B, L)
        loss_perfect = loss_fn(log_probs_perfect, targets, mask)
        loss_uniform = loss_fn(log_probs_uniform, targets, mask)
        assert loss_perfect.item() < loss_uniform.item()

    def test_uniform_predictions_loss_near_log_vocab(self) -> None:
        """Uniform predictions give loss approximately log(21)."""
        loss_fn = LabelSmoothedNLLLoss(label_smoothing=0.1, vocab_size=21)
        B, L, V = 2, 10, 21
        targets = torch.randint(0, V, (B, L))
        log_probs = torch.full((B, L, V), math.log(1.0 / V))
        mask = torch.ones(B, L)

        loss = loss_fn(log_probs, targets, mask)
        expected = math.log(V)
        assert abs(loss.item() - expected) < 0.1

    def test_masked_positions_excluded(self) -> None:
        """Masked positions don't contribute to loss."""
        loss_fn = LabelSmoothedNLLLoss(label_smoothing=0.1, vocab_size=21)
        B, L, V = 1, 10, 21
        targets = torch.randint(0, V, (B, L))
        log_probs = torch.randn(B, L, V)
        log_probs = torch.log_softmax(log_probs, dim=-1)

        # All masked → should not crash (denominator clamped to 1)
        mask_none = torch.zeros(B, L)
        loss_none = loss_fn(log_probs, targets, mask_none)
        assert loss_none.item() == 0.0

        # Partial mask: loss should differ from full mask
        mask_full = torch.ones(B, L)
        mask_partial = torch.zeros(B, L)
        mask_partial[0, :3] = 1.0
        loss_full = loss_fn(log_probs, targets, mask_full)
        loss_partial = loss_fn(log_probs, targets, mask_partial)
        assert not torch.isclose(loss_full, loss_partial)

    def test_gradient_flows(self) -> None:
        """Gradient flows through log_probs."""
        loss_fn = LabelSmoothedNLLLoss(label_smoothing=0.1, vocab_size=21)
        B, L, V = 1, 5, 21
        targets = torch.randint(0, V, (B, L))
        log_probs = torch.randn(B, L, V, requires_grad=True)
        mask = torch.ones(B, L)

        loss = loss_fn(torch.log_softmax(log_probs, dim=-1), targets, mask)
        loss.backward()
        assert log_probs.grad is not None
        assert not torch.all(log_probs.grad == 0)

    def test_no_smoothing_matches_nll(self) -> None:
        """With eps=0, loss should match standard NLL."""
        loss_fn = LabelSmoothedNLLLoss(label_smoothing=0.0, vocab_size=21)
        B, L, V = 2, 8, 21
        targets = torch.randint(0, V, (B, L))
        log_probs = torch.log_softmax(torch.randn(B, L, V), dim=-1)
        mask = torch.ones(B, L)

        loss = loss_fn(log_probs, targets, mask)
        # Manual NLL
        nll = -torch.gather(log_probs, 2, targets.unsqueeze(-1)).squeeze(-1)
        expected = nll.mean()
        assert torch.allclose(loss, expected, atol=1e-5)
