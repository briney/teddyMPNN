# teddyMPNN Simplification Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strip teddyMPNN down to a focused ProteinMPNN fine-tune on teddymer with interface-weighted loss, removing LigandMPNN, the NVIDIA data source, the benchmark harness, and weight-export machinery.

**Architecture:** In-place surgical trim on branch `refactor/simplify`. Delete out-of-scope code and its references first (keeping the test suite green after each task), then add the one net-new behavior — interface-weighted cross-entropy — via TDD. The validated ProteinMPNN model and the `identify_interface_residues` / `LabelSmoothedNLLLoss` primitives are reused, not rewritten.

**Tech Stack:** Python 3.11+, PyTorch, Pydantic + OmegaConf (config), Typer (CLI), pytest, ruff, mypy.

## Global Constraints

- Python 3.11+; line length 100; Google-style docstrings; type hints on all signatures (copied from CLAUDE.md).
- After every task: `pytest` green, `ruff check src/ tests/` clean, `ruff format src/ tests/` applied, `mypy src/` clean.
- Commit message format: `<component>: <what changed and why>`.
- **Regression invariant:** `interface_weight = 1.0` MUST produce loss numerically identical to the current standard CE. This is the safety net for the whole refactor.
- The Foundry-equivalence test for the ProteinMPNN model (`tests/validation/test_foundry_equivalence.py`) must continue to pass — keep its weight-load + forward-parity assertions; only drop any export round-trip assertions.
- Editable install is required for the suite: run `pip install -e ".[dev,data,train]"` once before starting.
- Line numbers below are from the pre-refactor tree (commit `dfbf76e`) and will drift as tasks land — anchor each edit on the quoted code, not the line number alone.

---

## Task 0: Establish green baseline

**Files:** none (verification only).

- [ ] **Step 1: Confirm install + baseline**

Run: `pip install -e ".[dev,data,train]" && pytest -q`
Expected: all tests pass (or note any pre-existing failures/skips — slow tests needing GPU/weights may skip). Record the passing count; every later task must not reduce it except for tests intentionally deleted.

- [ ] **Step 2: Confirm lint/type baseline**

Run: `ruff check src/ tests/ && mypy src/`
Expected: clean (or record pre-existing issues so they aren't attributed to this work).

---

## Task 1: Remove the multi-model benchmark harness

**Files:**
- Delete: `src/teddympnn/evaluation/benchmark.py`
- Delete: `scripts/run_benchmark.py`
- Delete: `configs/benchmark.yaml`
- Delete: `tests/evaluation/test_benchmark.py`
- Modify: `src/teddympnn/evaluation/__init__.py`
- Modify: `src/teddympnn/cli.py` (remove `evaluate benchmark` command, ~lines 283-339)
- Modify: `tests/test_cli.py` (remove benchmark command tests)

**Interfaces:**
- Produces: `teddympnn.evaluation` no longer exports `BenchmarkReport`, `BenchmarkResult`, `run_benchmark`.

- [ ] **Step 1: Delete the standalone files**

```bash
git rm src/teddympnn/evaluation/benchmark.py scripts/run_benchmark.py configs/benchmark.yaml tests/evaluation/test_benchmark.py
```

- [ ] **Step 2: Drop benchmark exports from `evaluation/__init__.py`**

Remove the import line `from teddympnn.evaluation.benchmark import BenchmarkReport, BenchmarkResult, run_benchmark` and the three names `"BenchmarkReport"`, `"BenchmarkResult"`, `"run_benchmark"` from `__all__`. Resulting file:

```python
"""Evaluation metrics for teddyMPNN models."""

from __future__ import annotations

from teddympnn.evaluation.binding_affinity import (
    predict_ddg,
    score_complex,
    score_structure,
)
from teddympnn.evaluation.sequence_recovery import RecoveryResults, compute_recovery
from teddympnn.evaluation.skempi import SKEMPIResults, evaluate_skempi

__all__ = [
    "RecoveryResults",
    "SKEMPIResults",
    "compute_recovery",
    "evaluate_skempi",
    "predict_ddg",
    "score_complex",
    "score_structure",
]
```

- [ ] **Step 3: Remove the `evaluate benchmark` CLI command**

In `src/teddympnn/cli.py`, delete the entire `@evaluate_app.command()` function named `benchmark` (the block at ~lines 283-339). Remove any now-unused imports it introduced (e.g. `run_benchmark`). Leave `recovery` and `ddg` commands intact.

- [ ] **Step 4: Remove benchmark tests from `tests/test_cli.py`**

Delete any test functions that invoke the benchmark command or import benchmark symbols. Search and verify none remain:

Run: `grep -rni "benchmark" src tests scripts configs`
Expected: no matches (benchmark fully removed).

- [ ] **Step 5: Verify + commit**

Run: `pytest -q && ruff check src/ tests/ && mypy src/`
Expected: green.

```bash
ruff format src/ tests/
git add -A
git commit -m "evaluation: remove multi-model benchmark harness"
```

---

## Task 2: Remove the NVIDIA data source

**Files:**
- Delete: `src/teddympnn/data/nvidia_complexes.py`
- Delete: `tests/data/test_nvidia_complexes.py`
- Modify: `src/teddympnn/config.py` (`SourceType` literal)
- Modify: `src/teddympnn/data/splits.py` (delete `split_nvidia_manifest`, remove its use in `prepare_manifests`)
- Modify: `src/teddympnn/cli.py` (delete `download nvidia-complexes`, remove nvidia handling in `prepare-manifests`)
- Modify: `configs/train.yaml` (remove nvidia source blocks)
- Modify: `tests/data/test_splits.py`, `tests/test_cli.py`, `tests/data/test_dataset.py`, `tests/evaluation/test_sequence_recovery.py` (drop nvidia references)

**Interfaces:**
- Produces: `SourceType = Literal["teddymer", "pdb"]`. `prepare_manifests` no longer accepts/uses an NVIDIA manifest argument.

- [ ] **Step 1: Delete standalone files**

```bash
git rm src/teddympnn/data/nvidia_complexes.py tests/data/test_nvidia_complexes.py
```

- [ ] **Step 2: Narrow `SourceType` in `config.py`**

Change line 29 from `SourceType = Literal["teddymer", "nvidia", "pdb"]` to:

```python
SourceType = Literal["teddymer", "pdb"]
```

- [ ] **Step 3: Remove NVIDIA from `data/splits.py`**

Delete the entire `def split_nvidia_manifest(...)` function (~lines 113-159). In `prepare_manifests`, delete the `if nvidia_manifest is not None:` block (~lines 349-363) and remove the `nvidia_manifest` parameter from the signature. Verify no other `nvidia` token remains in the file.

- [ ] **Step 4: Remove NVIDIA from `cli.py`**

Delete the `@download_app.command("nvidia-complexes")` function (~lines 91-118). In the `prepare-manifests` command, remove any `nvidia`-related option/argument and its pass-through to `prepare_manifests`.

- [ ] **Step 5: Remove NVIDIA from `configs/train.yaml`**

Delete the `nvidia:` block under `data.train` and the `nvidia:` block under `data.validation`.

- [ ] **Step 6: Purge nvidia references from tests**

Edit `tests/data/test_splits.py` (remove `split_nvidia_manifest` tests + nvidia args), `tests/test_cli.py` (remove nvidia-complexes command tests), `tests/data/test_dataset.py` and `tests/evaluation/test_sequence_recovery.py` (remove nvidia source references). Then:

Run: `grep -rni "nvidia" src tests scripts configs`
Expected: no matches.

- [ ] **Step 7: Verify + commit**

Run: `pytest -q && ruff check src/ tests/ && mypy src/`
Expected: green.

```bash
ruff format src/ tests/
git add -A
git commit -m "data: remove NVIDIA complexes data source"
```

---

## Task 3: Remove LigandMPNN (model + all model selection)

This task removes the `LigandMPNN` class and everything that imports or selects it. The dormant ligand *data* branches (`include_ligand_atoms` etc., which do not import the class) are left in place and removed in Task 4 — this keeps imports resolvable and the suite green.

**Files:**
- Delete: `src/teddympnn/models/ligand_mpnn.py`
- Delete: `tests/models/test_ligand_mpnn.py`
- Modify: `src/teddympnn/models/__init__.py`
- Modify: `src/teddympnn/config.py`
- Modify: `src/teddympnn/training/trainer.py`
- Modify: `src/teddympnn/evaluation/binding_affinity.py`
- Modify: `src/teddympnn/cli.py` (model_type options in `score`, `evaluate recovery`, `evaluate ddg`)
- Modify: `src/teddympnn/evaluation/sequence_recovery.py`, `src/teddympnn/evaluation/skempi.py`, `src/teddympnn/evaluation/_batch.py` (model selection → ProteinMPNN)
- Modify: `tests/test_config.py`, `tests/test_cli.py`, `tests/training/test_e2e_training.py`, `tests/evaluation/*` (drop ligand_mpnn cases)

**Interfaces:**
- Produces: `ModelType = Literal["protein_mpnn"]`. The only model class is `ProteinMPNN`. `TrainingConfig` no longer has `num_context_atoms`, `atomize_partner_sidechains`, `sidechain_atomization_probability`, `sidechain_atomization_per_residue_probability`.

- [ ] **Step 1: Delete the model + its test**

```bash
git rm src/teddympnn/models/ligand_mpnn.py tests/models/test_ligand_mpnn.py
```

- [ ] **Step 2: `models/__init__.py`**

```python
"""teddyMPNN model implementations."""

from __future__ import annotations

from teddympnn.models.protein_mpnn import ProteinMPNN

__all__ = ["ProteinMPNN"]
```

- [ ] **Step 3: `config.py` — remove ligand from the schema**

Make these edits:
- Delete the import `from teddympnn.models.ligand_mpnn import LigandMPNN` (line 24).
- `ModelType = Literal["protein_mpnn", "ligand_mpnn"]` → `ModelType = Literal["protein_mpnn"]`.
- `_MODEL_TYPE_TO_CLASS`: drop the `"ligand_mpnn": LigandMPNN,` entry.
- `_MODEL_TYPE_TRAINING_DEFAULTS`: drop the entire `"ligand_mpnn": {...}` entry.
- In `ModelConfig`: delete the field `num_context_atoms: int | None = None  # ligand_mpnn only`.
- In `TrainingConfig`: delete fields `atomize_partner_sidechains`, `sidechain_atomization_probability`, `sidechain_atomization_per_residue_probability`.
- In `apply_model_defaults`: delete `is_ligand = ...`, the `num_context_atoms` validation block, and the `if is_ligand and self.model.num_context_atoms is None:` block. The validator's model-defaults region becomes:

```python
        model_cls = _MODEL_TYPE_TO_CLASS[self.model_type]

        # Architecture defaults from the model class's __init__ signature.
        arch_fields = (
            "hidden_dim",
            "num_encoder_layers",
            "num_decoder_layers",
            "num_neighbors",
            "dropout",
        )
        for field in arch_fields:
            if getattr(self.model, field) is None:
                setattr(self.model, field, _model_init_default(model_cls, field))

        # Training knob defaults per model_type.
        defaults = _MODEL_TYPE_TRAINING_DEFAULTS[self.model_type]
```

(Leave the `token_budget`/`structure_noise`/`grad_clip_max_norm` and pretrained-weights defaulting below it unchanged.)

- [ ] **Step 4: `trainer.py` — hardcode ProteinMPNN, drop ligand args**

- Replace the import of `LigandMPNN` (and `ProteinMPNN` if combined) so only `ProteinMPNN` is imported.
- Replace `is_ligand = config.model_type == "ligand_mpnn"` / `model_cls = LigandMPNN if is_ligand else ProteinMPNN` (~lines 165-166) with `model_cls = ProteinMPNN`.
- Delete the `if is_ligand:` block (~line 174).
- In both `PPIDataset(...)` constructions (~lines 188-192 and ~218-222), remove the `include_ligand_atoms=...` and `atomize_partner_sidechains=...` keyword arguments (keep `min_interface_contacts=config.min_interface_contacts`).

- [ ] **Step 5: `evaluation/binding_affinity.py`**

- Delete `from teddympnn.models.ligand_mpnn import LigandMPNN` (line 26).
- Simplify the `_model_type` helper (~lines 48-49) to return `"protein_mpnn"` unconditionally, or inline-remove it and use `ProteinMPNN` directly at its call sites.

- [ ] **Step 6: `cli.py` + remaining eval entry points — drop model_type selection**

In `cli.py` `score`, `evaluate recovery`, and `evaluate ddg` commands: remove the `model_type` Typer option, replace each `model_cls = LigandMPNN if model_type == "ligand_mpnn" else ProteinMPNN` with `model_cls = ProteinMPNN`, remove the `include_ligand_atoms=(model_type == "ligand_mpnn")` argument from the `PPIDataset(...)` call in `recovery` (~line 259), and delete the `if model_type == "ligand_mpnn":` ligand-atom block in `score` (~lines 443-460). Apply the same ProteinMPNN-only simplification to any `model_type`/ligand branch in `evaluation/sequence_recovery.py`, `evaluation/skempi.py`, and `evaluation/_batch.py`.

- [ ] **Step 7: Purge ligand from tests**

Edit `tests/test_config.py` (remove `ligand_mpnn` model_type + `num_context_atoms` cases), `tests/test_cli.py` (remove ligand model_type options), `tests/training/test_e2e_training.py` and any `tests/evaluation/*` that reference ligand. Then confirm only intentional, ProteinMPNN-shared references remain:

Run: `grep -rni "ligandmpnn\|ligand_mpnn\|num_context_atoms\|atomize_partner" src tests`
Expected: no matches in `src/`; no matches in `tests/`. (Generic `ligand`-atom data code in `features.py`/`dataset.py`/`collator.py` is removed in Task 4.)

- [ ] **Step 8: Verify + commit**

Run: `pytest -q && ruff check src/ tests/ && mypy src/`
Expected: green.

```bash
ruff format src/ tests/
git add -A
git commit -m "models: drop LigandMPNN and all model-type selection"
```

---

## Task 4: Remove dormant ligand data branches + dead element constants

With no model consuming ligand atoms, the ligand featurization, dataset context branches, and collator ligand-padding are dead. Removing them shrinks `features.py`, `dataset.py`, and `collator.py`.

**Files:**
- Modify: `src/teddympnn/data/features.py` (delete `extract_ligand_atoms`, `extract_sidechain_atoms`, `EXCLUDED_LIGAND_RESIDUES`, `EXCLUDED_IONS`, and the now-dead element constants `_ELEMENTS`/`NUM_ELEMENT_TYPES`/`UNK_ELEMENT_IDX`/`element_to_idx`)
- Modify: `src/teddympnn/data/dataset.py` (drop `include_ligand_atoms`, `atomize_partner_sidechains`, `sidechain_atomization_*` params and the Y-context block)
- Modify: `src/teddympnn/data/collator.py` (drop ligand keys/padding)
- Modify: `src/teddympnn/data/__init__.py` (drop `extract_ligand_atoms` export)
- Modify: `src/teddympnn/data/sampler.py` (remove LigandMPNN mention in comment)
- Modify: `tests/data/test_features.py`, `tests/data/test_dataset.py`, `tests/data/test_collator.py`

**Interfaces:**
- Produces: `PPIDataset.__getitem__` returns no `Y`/`Y_m`/`Y_t` keys. `PPIDataset.__init__` signature drops the four ligand/atomization parameters. `PaddingCollator` no longer has `LIGAND_KEYS`.

- [ ] **Step 1: `features.py` — delete ligand functions + dead constants**

Delete `def extract_ligand_atoms(...)` (~lines 388-446) and `def extract_sidechain_atoms(...)` (~lines 483-541), the `EXCLUDED_LIGAND_RESIDUES`/`EXCLUDED_IONS` constants (~lines 175-177), and the element-type constants `_ELEMENTS`/`NUM_ELEMENT_TYPES`/`UNK_ELEMENT_IDX`/`element_to_idx` (~lines 51-173) — verify via grep these constants are referenced only by the two deleted functions before removing. Keep `derive_backbone`, `parse_structure`, `identify_interface_residues`, `_compute_cb`, and `NUM_ATOMS_37`/`ATOM_ORDER`.

Run: `grep -n "element_to_idx\|UNK_ELEMENT_IDX\|_ELEMENTS\|EXCLUDED_" src/teddympnn/data/features.py`
Expected: no matches after deletion.

- [ ] **Step 2: `dataset.py` — drop ligand params and context block**

- Remove `include_ligand_atoms`, `atomize_partner_sidechains`, `sidechain_atomization_probability`, `sidechain_atomization_per_residue_probability` from `PPIDataset.__init__` and their `self.*` assignments.
- In `_load_features`, delete the `if self.include_ligand_atoms: ...` block (~lines 232-235).
- In `__getitem__`, delete the entire ligand-context section (~lines 288-323, the `if self.include_ligand_atoms ... else ...` that sets `result["Y"]`/`Y_m`/`Y_t`). The dict `result` ends with `num_residues`/`source` (interface mask is added in Task 7).
- Remove the now-unused imports `extract_ligand_atoms`, `extract_sidechain_atoms`, and `random` if no longer used.

- [ ] **Step 3: `collator.py` — drop ligand keys/padding**

- Remove `"Y"`, `"Y_m"`, `"Y_t"` from `_PAD_VALUES`.
- Delete the `LIGAND_KEYS` frozenset.
- Remove the `N_max = ...` line and the `elif key in self.LIGAND_KEYS:` branch (the whole ligand-padding block). The `__call__` loop becomes residue-keys / scalar-tensor-stack / metadata-list only:

```python
        B = len(batch)
        result: dict[str, Any] = {}

        L_max = max(b["S"].shape[0] for b in batch)

        for key in batch[0]:
            if key in self.RESIDUE_KEYS:
                result[key] = self._pad_and_stack(batch, key, L_max, dim=0)
            elif isinstance(batch[0][key], torch.Tensor):
                result[key] = torch.stack([b[key] for b in batch])
            else:
                result[key] = [b[key] for b in batch]

        return result
```

- [ ] **Step 4: `data/__init__.py` + `sampler.py`**

Remove `extract_ligand_atoms` from the import and `__all__` in `data/__init__.py`. In `sampler.py`, reword the docstring comment that mentions LigandMPNN's recommended token budget to be model-agnostic.

- [ ] **Step 5: Update data tests**

Remove ligand-atom assertions/parameters from `tests/data/test_features.py`, `tests/data/test_dataset.py`, and `tests/data/test_collator.py`.

Run: `grep -rni "ligand\|extract_sidechain\|\"Y\"\|'Y'\|Y_m\|Y_t" src/teddympnn/data tests/data`
Expected: only legitimate non-ligand matches (review each; there should be none referencing ligand context).

- [ ] **Step 6: Verify + commit**

Run: `pytest -q && ruff check src/ tests/ && mypy src/`
Expected: green.

```bash
ruff format src/ tests/
git add -A
git commit -m "data: remove dead ligand featurization and collation"
```

---

## Task 5: Trim weight-export machinery

Keep all load paths; remove export-to-foreign-format paths and the export CLI command.

**Files:**
- Modify: `src/teddympnn/weights/foundry.py` (delete `export_foundry_checkpoint`)
- Modify: `src/teddympnn/weights/legacy.py` (delete `convert_to_legacy`, `_restore_120th_atom_type`, and now-unused `current_to_legacy_*` permutation imports)
- Modify: `src/teddympnn/cli.py` (delete `checkpoints export-foundry`; remove the `checkpoints` Typer group if now empty)
- Modify: `tests/weights/test_foundry.py`, `tests/weights/test_legacy.py` (drop export tests; keep load tests)
- Modify: `tests/validation/test_foundry_equivalence.py` (keep load + forward-parity; drop any export round-trip)

**Interfaces:**
- Produces: `weights.foundry` exposes only `load_foundry_checkpoint`; `weights.legacy` exposes only `load_legacy_weights` (and its private `_drop_120th_atom_type` helper). `weights.io` is unchanged (native save/load kept).

- [ ] **Step 1: `foundry.py`**

Delete `def export_foundry_checkpoint(...)` (~lines 54-79). Keep `load_foundry_checkpoint`.

- [ ] **Step 2: `legacy.py`**

Delete `def convert_to_legacy(...)` (~lines 283-317) and `def _restore_120th_atom_type(...)` (~lines 197-219). Keep `load_legacy_weights` and `_drop_120th_atom_type`. Remove now-unused imports from `teddympnn.models.tokens` (`current_to_legacy_rbf_permutation`, `current_to_legacy_token_permutation`) — verify with grep they're unused before removing; keep the `legacy_to_current_*` and `expand_pair_permutation` imports that the load path needs.

- [ ] **Step 3: `cli.py`**

Delete the `@checkpoints_app.command("export-foundry")` function (~lines 204-224) and its `export_foundry_checkpoint` import. If the `checkpoints` Typer sub-app now has no commands, remove its definition and its `app.add_typer(...)` registration.

- [ ] **Step 4: Tests**

In `tests/weights/test_foundry.py` remove tests of `export_foundry_checkpoint`; keep load tests. In `tests/weights/test_legacy.py` remove tests of `convert_to_legacy`/`_restore_120th_atom_type`; keep `load_legacy_weights` tests. Open `tests/validation/test_foundry_equivalence.py` and confirm it still asserts (a) base weights load into `ProteinMPNN` and (b) forward outputs match the reference; remove only any export/round-trip assertion. Then:

Run: `grep -rni "export_foundry\|convert_to_legacy\|_restore_120th" src tests`
Expected: no matches.

- [ ] **Step 5: Verify + commit**

Run: `pytest -q && ruff check src/ tests/ && mypy src/`
Expected: green (Foundry-equivalence test still passing).

```bash
ruff format src/ tests/
git add -A
git commit -m "weights: drop Foundry/legacy export, keep load paths"
```

---

## Task 6: Extend the loss with optional per-residue weights (TDD)

**Files:**
- Modify: `src/teddympnn/training/loss.py`
- Test: `tests/training/test_loss.py`

**Interfaces:**
- Produces: `LabelSmoothedNLLLoss.forward(log_probs, targets, mask, weights=None)` where `weights: torch.Tensor | None` has shape `(B, L)`. The loss is a weighted mean: `Σ(nll · mask · w) / Σ(mask · w)`. `weights=None` and any constant `weights` reproduce the unweighted result.

- [ ] **Step 1: Write the failing tests**

Add to `tests/training/test_loss.py`:

```python
import torch

from teddympnn.training.loss import LabelSmoothedNLLLoss


def test_weights_none_matches_unweighted():
    torch.manual_seed(0)
    b, length, vocab = 2, 5, 21
    log_probs = torch.log_softmax(torch.randn(b, length, vocab), dim=-1)
    targets = torch.randint(0, vocab, (b, length))
    mask = torch.ones(b, length)
    loss_fn = LabelSmoothedNLLLoss()
    unweighted = loss_fn(log_probs, targets, mask)
    explicit_ones = loss_fn(log_probs, targets, mask, weights=torch.ones(b, length))
    assert torch.allclose(unweighted, explicit_ones)


def test_constant_weight_is_invariant():
    torch.manual_seed(1)
    b, length, vocab = 2, 4, 21
    log_probs = torch.log_softmax(torch.randn(b, length, vocab), dim=-1)
    targets = torch.randint(0, vocab, (b, length))
    mask = torch.ones(b, length)
    loss_fn = LabelSmoothedNLLLoss()
    base = loss_fn(log_probs, targets, mask)
    scaled = loss_fn(log_probs, targets, mask, weights=torch.full((b, length), 3.0))
    assert torch.allclose(base, scaled)


def test_upweighting_high_loss_positions_increases_loss():
    b, length, vocab = 1, 3, 21
    logits = torch.full((b, length, vocab), -10.0)
    targets = torch.tensor([[0, 1, 2]])
    logits[0, 0, 5] = 10.0  # predicts 5, target 0 -> high NLL at position 0
    logits[0, 1, 1] = 10.0  # correct
    logits[0, 2, 2] = 10.0  # correct
    log_probs = torch.log_softmax(logits, dim=-1)
    mask = torch.ones(b, length)
    loss_fn = LabelSmoothedNLLLoss(label_smoothing=0.0)
    base = loss_fn(log_probs, targets, mask)
    up = loss_fn(log_probs, targets, mask, weights=torch.tensor([[5.0, 1.0, 1.0]]))
    assert up > base
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/training/test_loss.py -q`
Expected: FAIL — `forward()` got an unexpected keyword argument `weights`.

- [ ] **Step 3: Implement the weighted loss**

Replace `LabelSmoothedNLLLoss.forward` in `src/teddympnn/training/loss.py` with:

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/training/test_loss.py -q`
Expected: PASS (including pre-existing loss tests).

- [ ] **Step 5: Commit**

```bash
ruff format src/ tests/
git add src/teddympnn/training/loss.py tests/training/test_loss.py
git commit -m "training: support per-residue weights in NLL loss"
```

---

## Task 7: Wire interface-weighted CE through data → trainer

**Files:**
- Modify: `src/teddympnn/data/dataset.py` (add `interface_residue_mask` to each example)
- Modify: `src/teddympnn/data/collator.py` (pad `interface_residue_mask`)
- Modify: `src/teddympnn/config.py` (add `interface_weight` knob)
- Modify: `src/teddympnn/training/trainer.py` (build weights, pass to loss; use batched interface mask in validation)
- Modify: `configs/train.yaml` (document `interface_weight`)
- Test: `tests/data/test_dataset.py`, `tests/data/test_collator.py`, `tests/training/test_e2e_training.py`

**Interfaces:**
- Consumes: `LabelSmoothedNLLLoss.forward(..., weights=...)` from Task 6; `identify_interface_residues(xyz_37, xyz_37_m, chain_labels)` from `data.features`.
- Produces: every batch carries `interface_residue_mask` of shape `(B, L)` (bool). `TrainingConfig.interface_weight: float = 1.0`. Trainer applies `weight = 1 + (interface_weight - 1) * interface_mask` to designed positions.

- [ ] **Step 1: Write failing tests**

In `tests/data/test_dataset.py`, add an assertion (reuse the existing dataset fixture) that a sample contains the interface mask:

```python
def test_getitem_includes_interface_mask(ppi_dataset):
    sample = ppi_dataset[0]
    mask = sample["interface_residue_mask"]
    assert mask.dtype == torch.bool
    assert mask.shape == sample["S"].shape
```

In `tests/data/test_collator.py`, add `interface_residue_mask` to the per-example dicts the test builds and assert it is padded along L:

```python
def test_collate_pads_interface_mask(collator):
    a = _make_example(length=4)
    b = _make_example(length=6)
    a["interface_residue_mask"] = torch.ones(4, dtype=torch.bool)
    b["interface_residue_mask"] = torch.zeros(6, dtype=torch.bool)
    out = collator([a, b])
    assert out["interface_residue_mask"].shape == (2, 6)
    assert out["interface_residue_mask"].dtype == torch.bool
    assert out["interface_residue_mask"][0, 4:].sum() == 0  # padded with False
```

(Adapt `ppi_dataset`/`collator`/`_make_example` to the existing fixtures/helpers in those files.)

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/data/test_dataset.py::test_getitem_includes_interface_mask tests/data/test_collator.py::test_collate_pads_interface_mask -q`
Expected: FAIL — `KeyError: 'interface_residue_mask'`.

- [ ] **Step 3: Add interface mask in `dataset.py`**

Ensure `identify_interface_residues` is imported from `teddympnn.data.features`. In `__getitem__`, immediately before `return result`, insert:

```python
        # Per-residue interface mask (computed on valid residues, scattered back).
        res_mask = features["residue_mask"].bool()
        interface_residue_mask = torch.zeros(L, dtype=torch.bool)
        if res_mask.any():
            interface_residue_mask[res_mask] = identify_interface_residues(
                features["xyz_37"][res_mask],
                features["xyz_37_m"][res_mask],
                features["chain_labels"][res_mask],
            )
        result["interface_residue_mask"] = interface_residue_mask
```

- [ ] **Step 4: Pad it in `collator.py`**

Add `"interface_residue_mask": False` to `_PAD_VALUES` and `"interface_residue_mask"` to the `RESIDUE_KEYS` frozenset.

- [ ] **Step 5: Run data tests to verify pass**

Run: `pytest tests/data/test_dataset.py tests/data/test_collator.py -q`
Expected: PASS.

- [ ] **Step 6: Add the config knob**

In `src/teddympnn/config.py` `TrainingConfig`, add after `label_smoothing`:

```python
    interface_weight: float = 1.0  # CE weight multiplier for interface residues (1.0 = standard)
```

- [ ] **Step 7: Apply weighting in `trainer.py`**

In `train_step`, replace the loss call (~lines 297-303) with:

```python
            output = self.model(batch)
            weights = 1.0 + (self.config.interface_weight - 1.0) * batch[
                "interface_residue_mask"
            ].float()
            loss = self.loss_fn(
                output["log_probs"],
                batch["S"],
                batch["designed_residue_mask"],
                weights=weights,
            )
```

In `validate`, apply the same `weights` construction to the validation loss call (~lines 340-345), and replace the on-the-fly interface recompute (~lines 362-369) with the batched mask:

```python
                full_interface = batch["interface_residue_mask"][b].bool()
                designed_interface = designed & full_interface
```

Remove the now-unused `identify_interface_residues` import from `trainer.py` if nothing else uses it.

- [ ] **Step 8: Document the knob in `configs/train.yaml`**

Add `interface_weight: 1.0` near `label_smoothing` with a short comment noting `>1.0` upweights interface residues.

- [ ] **Step 9: Extend the e2e training test**

In `tests/training/test_e2e_training.py`, set `interface_weight=3.0` (via config override or the config object) in the pilot run and assert training completes the configured steps with a finite loss. Keep an assertion (or add one) that with `interface_weight=1.0` the run still trains — this guards the regression invariant end to end.

Run: `pytest tests/training/test_e2e_training.py -q`
Expected: PASS.

- [ ] **Step 10: Verify + commit**

Run: `pytest -q && ruff check src/ tests/ && mypy src/`
Expected: green.

```bash
ruff format src/ tests/
git add -A
git commit -m "training: interface-weighted cross-entropy via config knob"
```

---

## Task 8: Documentation + README cleanup

**Files:**
- Delete: `docs/WORKPLAN.md`, `docs/TECHNICAL_ANALYSIS.md`
- Modify: `docs/ARCHITECTURE.md` (shrink to current scope, point to `docs/VISION.md`)
- Modify: `README.md` (remove benchmark/ligand/nvidia/export examples; reflect 3-command CLI + `interface_weight`)
- Modify: `CLAUDE.md` (fill the Architecture section briefly; update install extras if changed)

- [ ] **Step 1: Delete stale design docs**

```bash
git rm docs/WORKPLAN.md docs/TECHNICAL_ANALYSIS.md
```

- [ ] **Step 2: Shrink ARCHITECTURE.md**

Reduce `docs/ARCHITECTURE.md` to the current system: ProteinMPNN model, teddymer data pipeline, interface-weighted training, recovery + SKEMPI evaluation. Remove all LigandMPNN, NVIDIA, benchmark, and Foundry-export content. Add a top line pointing to `docs/VISION.md` as the north star.

- [ ] **Step 3: Update README.md**

Remove the benchmark Quick Start example and any LigandMPNN/NVIDIA/export references. Ensure the CLI examples cover only `train`, `score`, and `evaluate {recovery,ddg}`. Mention `interface_weight` as the interface-emphasis knob.

- [ ] **Step 4: Update CLAUDE.md**

Replace the `## Architecture` "TODO" with a 3-4 sentence summary matching VISION. Confirm the install extras line matches `pyproject.toml`.

- [ ] **Step 5: Consistency sweep + commit**

Run: `grep -rni "ligand\|nvidia\|benchmark\|export-foundry\|convert_to_legacy" docs README.md CLAUDE.md`
Expected: no stale references (any remaining `ligand` mention must be deliberate and accurate).

```bash
git add -A
git commit -m "docs: refocus README and architecture docs on the fine-tune"
```

---

## Task 9: Final dead-code & verification sweep

**Files:** whole repo (verification + any residual cleanup).

- [ ] **Step 1: Grep for orphaned references**

Run:
```bash
grep -rni "LigandMPNN\|ligand_mpnn\|nvidia\|benchmark\|export_foundry\|convert_to_legacy\|num_context_atoms\|atomize_partner\|extract_ligand\|extract_sidechain" src tests scripts configs
```
Expected: no matches. Fix any stragglers (and commit per the relevant component).

- [ ] **Step 2: Confirm CLI surface**

Run: `python -m teddympnn --help` and the subcommand helps.
Expected: commands are `train`, `score`, `evaluate` (with `recovery`, `ddg`), and the retained `download` helpers — no `benchmark`, `nvidia-complexes`, or `export-foundry`.

- [ ] **Step 3: Full verification**

Run: `ruff format --check src/ tests/ && ruff check src/ tests/ && mypy src/ && pytest -q`
Expected: all clean/green. The Foundry-equivalence test passes; `test_e2e_training.py` trains the pilot model with interface weighting.

- [ ] **Step 4: Final commit (if any residual fixes)**

```bash
git add -A
git commit -m "chore: final dead-code sweep for simplification refactor"
```

---

## Self-Review

**Spec coverage** (against `docs/VISION.md`):
- Interface sequence design fine-tune → Tasks 6-7 (interface-weighted CE) + retained ProteinMPNN/teddymer pipeline. ✓
- ddG / SKEMPI eval kept → preserved in Tasks 1/3/5 (skempi + binding_affinity retained, only ligand/benchmark stripped). ✓
- Drop LigandMPNN → Tasks 3-4. ✓
- Drop NVIDIA → Task 2. ✓
- Drop benchmark harness → Task 1. ✓
- Trim weight I/O (load-only) → Task 5. ✓
- teddymer train + held-out recovery + PDB recovery → data pipeline and `evaluate recovery` retained throughout; PDB loader untouched. ✓
- OmegaConf/Hydra overrides + 3-command CLI → Tasks 1-3/8 simplify CLI; config loader untouched. ✓
- Docs: VISION kept, WORKPLAN/TECHNICAL_ANALYSIS removed, ARCHITECTURE shrunk → Task 8. ✓
- Success criteria (beat base on interface recovery, no non-interface regression, ≥ match ddG) are measured by the retained `evaluate` commands after training — out of scope for the refactor code itself, but the tooling to measure them all survives. ✓

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N" — deletions reference exact files/symbols; the feature task includes complete test + implementation code. ✓

**Type consistency:** `interface_residue_mask` (bool `(B, L)`) is produced in Task 7 dataset/collator and consumed in Task 7 trainer; `weights` kwarg signature in Task 6 matches the trainer call in Task 7; `ModelType`/`SourceType` literals are narrowed once and referenced consistently. ✓
