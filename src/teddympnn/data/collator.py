"""Padding collator for variable-length protein structure batches.

Pads all tensors in a batch to the longest sequence, using appropriate
padding values for each feature type, and stacks into batched tensors.
"""

from __future__ import annotations

from typing import Any

import torch

# Padding values per feature key
_PAD_VALUES: dict[str, float | int | bool] = {
    "xyz_37": 0.0,
    "xyz_37_m": False,
    "X": 0.0,
    "X_m": False,
    "S": 20,  # UNK token index
    "R_idx": -100,
    "chain_labels": -1,
    "residue_mask": False,
    "designed_residue_mask": False,
    "fixed_residue_mask": False,
    "Y": 0.0,
    "Y_m": False,
    "Y_t": 0,
}


class PaddingCollator:
    """Collates variable-length structure feature dicts into padded batches.

    Pads protein residue tensors (keyed by residue length L) and ligand atom
    tensors (keyed by atom count N) to their respective maxima within the
    batch.

    Non-tensor values (e.g. ``num_residues``) are collected into lists.
    """

    # Keys that are padded along the residue (L) dimension
    RESIDUE_KEYS: frozenset[str] = frozenset(
        {
            "xyz_37",
            "xyz_37_m",
            "X",
            "X_m",
            "S",
            "R_idx",
            "chain_labels",
            "residue_mask",
            "designed_residue_mask",
            "fixed_residue_mask",
        }
    )

    # Keys that are padded along the ligand atom (N) dimension
    LIGAND_KEYS: frozenset[str] = frozenset({"Y", "Y_m", "Y_t"})

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Pad and stack a list of feature dicts into a batched dict.

        Args:
            batch: List of per-example feature dicts from ``PPIDataset``.

        Returns:
            Batched dict with shape ``(B, L_max, ...)`` for residue tensors,
            ``(B, N_max, ...)`` for ligand tensors, and lists for metadata.
        """
        if not batch:
            msg = "Cannot collate empty batch"
            raise ValueError(msg)

        B = len(batch)
        result: dict[str, Any] = {}

        # Find max lengths
        L_max = max(b["S"].shape[0] for b in batch)
        N_max = max(b["Y"].shape[0] for b in batch) if "Y" in batch[0] else 0

        for key in batch[0]:
            if key in self.RESIDUE_KEYS:
                result[key] = self._pad_and_stack(batch, key, L_max, dim=0)
            elif key in self.LIGAND_KEYS:
                if N_max > 0:
                    result[key] = self._pad_and_stack(batch, key, N_max, dim=0)
                else:
                    # All examples have empty ligand tensors — keep empty
                    sample = batch[0][key]
                    if sample.dim() == 1:
                        result[key] = torch.zeros(B, 0, dtype=sample.dtype)
                    else:
                        shape = [B, 0] + list(sample.shape[1:])
                        result[key] = torch.zeros(*shape, dtype=sample.dtype)
            elif isinstance(batch[0][key], torch.Tensor):
                # Scalar tensors — just stack
                result[key] = torch.stack([b[key] for b in batch])
            else:
                # Non-tensor metadata (e.g. num_residues, chain_ids)
                result[key] = [b[key] for b in batch]

        return result

    @staticmethod
    def _pad_and_stack(
        batch: list[dict[str, Any]],
        key: str,
        target_len: int,
        dim: int,
    ) -> torch.Tensor:
        """Pad tensors along the specified dimension and stack into a batch.

        Args:
            batch: List of feature dicts.
            key: Feature key to pad.
            target_len: Target length for padding.
            dim: Dimension to pad along.

        Returns:
            Batched tensor of shape ``(B, target_len, ...)``.
        """
        pad_value = _PAD_VALUES.get(key, 0)
        tensors: list[torch.Tensor] = []

        for b in batch:
            t = b[key]
            cur_len = t.shape[dim]
            pad_amount = target_len - cur_len

            if pad_amount == 0:
                tensors.append(t)
                continue

            # Build F.pad-style padding (reversed dimension order, innermost first)
            # For dim=0 of tensor with shape (L, ...), pad at the end of dim 0
            ndim = t.dim()
            pad_widths = [0] * (2 * ndim)
            # F.pad pairs go from last dim to first
            pad_idx = 2 * (ndim - 1 - dim) + 1
            pad_widths[pad_idx] = pad_amount

            t_pad = t.float() if isinstance(pad_value, float) else t
            padded = torch.nn.functional.pad(t_pad, pad_widths, value=pad_value)

            # Restore original dtype if we cast to float for padding
            if padded.dtype != t.dtype:
                padded = padded.to(t.dtype)

            tensors.append(padded)

        return torch.stack(tensors)
