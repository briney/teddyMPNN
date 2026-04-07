# teddyMPNN Work Plan

Detailed implementation plan for teddyMPNN. This plan assumes the architecture
described in [ARCHITECTURE.md](ARCHITECTURE.md): standalone runtime, checkpoint
parity, bidirectional partner design, separate dataset adapters, and a native
checkpoint bundle.

## Phase 1: Model Core And Compatibility

### 1.1 — Tokens And Compatibility Conventions

Define the compatibility constants that every later module depends on.

**Files:**

- `src/teddympnn/models/tokens.py`
- `tests/models/test_tokens.py`

**Implement:**

- current token order
- legacy token order
- ordered 25-pair backbone/virtual-atom RBF convention
- positional-encoding constants:
  - `max_relative_feature = 32`
  - `num_offset_bins = 65`
  - `inter_chain_bucket = 65`
  - `num_positional_classes = 66`
- helper permutations for:
  - legacy -> current token order
  - current -> legacy token order
  - legacy -> current RBF pair order
  - current -> legacy RBF pair order

**Acceptance:**

- token and pair-order round-trips are identity
- positional-encoding constants imply a 66-class projection everywhere

### 1.2 — Leaf Layers

Implement the reusable layers used by both model families.

**Files:**

- `src/teddympnn/models/layers/feed_forward.py`
- `src/teddympnn/models/layers/positional_encoding.py`
- `src/teddympnn/models/layers/message_passing.py`
- `src/teddympnn/models/layers/__init__.py`
- `tests/models/layers/test_feed_forward.py`
- `tests/models/layers/test_positional_encoding.py`
- `tests/models/layers/test_message_passing.py`

**Requirements:**

- `PositionWiseFeedForward` with reference-compatible parameter naming
- `PositionalEncodings` using `66` one-hot classes, with inter-chain bucket
  index `65`
- `EncLayer`
- `DecLayer`
- message-passing helpers:
  - `gather_nodes`
  - `gather_edges`
  - `cat_neighbors_nodes`

**Acceptance:**

- forward shapes are correct
- masks behave correctly
- gradients flow through all parameters

### 1.3 — Graph Embeddings

Implement graph featurization for the two model families.

**Files:**

- `src/teddympnn/models/layers/graph_embeddings.py`
- `tests/models/layers/test_graph_embeddings.py`

**ProteinFeatures:**

- Input contract uses `X` as backbone-only `(B, L, 4, 3)` tensor
- Compute virtual `CB`
- Build `CA` k-nearest-neighbor graph
- Produce ordered 25-pair RBF edge features plus positional features

**ProteinFeaturesLigand:**

- Reuse the ProteinMPNN backbone graph
- Consume:
  - `xyz_37`
  - `xyz_37_mask`
  - `Y`, `Y_m`, `Y_t`
- Do not invent a new partner encoder
- Assume partner side-chain atoms have already been merged into the `Y` context
  pool by upstream batch-building logic

**Acceptance:**

- dimensions match the compatibility contract
- nearest-neighbor indices are correct for known fixtures
- positional embedding input width matches `66`

### 1.4 — ProteinMPNN Core

Implement the checkpoint-compatible ProteinMPNN model core.

**Files:**

- `src/teddympnn/models/protein_mpnn.py`
- `tests/models/test_protein_mpnn.py`

**Requirements:**

- match reference module names and state-dict structure where portability
  depends on them
- support teacher-forcing scoring and autoregressive sampling
- consume the shared design-view masks:
  - `designed_residue_mask`
  - `fixed_residue_mask`
  - `mask_for_loss`

**Acceptance:**

- strict-load current compatible weights
- produce valid log-probabilities
- output shapes match the reference model family

### 1.5 — LigandMPNN Core

Implement the checkpoint-compatible LigandMPNN model core.

**Files:**

- `src/teddympnn/models/ligand_mpnn.py`
- `tests/models/test_ligand_mpnn.py`

**Requirements:**

- preserve the existing ligand-context pathway
- do not add a new protein-partner encoder
- consume partner and hetero-atom context through `Y / Y_m / Y_t`
- hide designed-partner side chains from the context path
- allow fixed-partner side-chain atoms to be exposed as context

**Acceptance:**

- strict-load compatible weights
- empty-context behavior is well-defined
- context path changes encoder outputs when valid context atoms are present

### 1.6 — Native Checkpoint Bundle And Compatibility Adapters

Implement checkpoint I/O around a native bundle, with explicit import/export
adapters.

**Files:**

- `src/teddympnn/weights/io.py`
- `src/teddympnn/weights/foundry.py`
- `src/teddympnn/weights/legacy.py`
- `tests/weights/test_io.py`
- `tests/weights/test_foundry.py`
- `tests/weights/test_legacy.py`

**`io.py`:**

- `save_checkpoint_bundle(...)`
- `load_checkpoint_bundle(...)`
- use native bundle fields:
  - `format_version`
  - `model_family`
  - `model_config`
  - `training_config`
  - `state_dict`
  - `optimizer_state`
  - `scheduler_state`
  - `step`
  - `metrics`
  - `compatibility`

**`foundry.py`:**

- import current Foundry-format checkpoints
- export Foundry-current strict-loadable checkpoints

**`legacy.py`:**

- import original legacy ProteinMPNN and LigandMPNN checkpoints
- export legacy-format checkpoints where supported
- apply token-order, RBF-pair-order, and atom-type transformations explicitly

**Acceptance:**

- legacy import -> native bundle -> Foundry export round-trip works
- current Foundry import -> native bundle -> Foundry export preserves strict
  loadability
- compatibility metadata records token order, RBF order, atom-type vocab size,
  and positional-encoding class count

### 1.7 — Validation Gate: Reference Equivalence

Verify the model cores against a reference implementation at the checkpoint and
tensor-output boundary.

**Files:**

- `tests/validation/test_foundry_equivalence.py`
- `tests/validation/conftest.py`
- `scripts/generate_foundry_reference.py`

**Approach:**

- generate reference tensors from the reference implementation
- compare:
  - neighbor indices
  - encoder outputs
  - log-probabilities
- validate both ProteinMPNN and LigandMPNN families

**Acceptance:**

- outputs match within fp32 tolerance
- this gates all later implementation phases

## Phase 2: Data Normalization And View Generation

### 2.1 — Structure Parsing Into `InterfaceExample`

Parse structures into the normalized structure-level contract.

**Files:**

- `src/teddympnn/data/contracts.py`
- `src/teddympnn/data/features.py`
- `tests/data/test_features.py`

**Implement:**

- `InterfaceExample` dataclass or model
- parsing of:
  - `xyz_37`
  - `xyz_37_mask`
  - `S`
  - `R_idx`
  - `chain_ids`
  - `chain_labels`
  - `residue_mask`
  - partner-group assignments
  - interface masks
  - real hetero-atom tables (`hetero_Y`, `hetero_Y_m`, `hetero_Y_t`)

**Acceptance:**

- a parsed example has one unambiguous meaning for every tensor
- `X` is not used at this layer

### 2.2 — Source-Specific Dataset Adapters

Normalize each source independently.

**Files:**

- `src/teddympnn/data/teddymer.py`
- `src/teddympnn/data/nvidia_complexes.py`
- `src/teddympnn/data/pdb_complexes.py`
- `tests/data/test_teddymer.py`
- `tests/data/test_nvidia_complexes.py`
- `tests/data/test_pdb_complexes.py`

**Requirements:**

- each adapter produces `InterfaceExample`
- preserve source provenance and quality metadata
- do not collapse source-specific assumptions into one monolithic parser

**Acceptance:**

- all adapters emit the same normalized contract
- provenance survives normalization

### 2.3 — `PartnerDesignView` Expansion

Expand normalized examples into the training views used by sequence-design
fine-tuning.

**Files:**

- `src/teddympnn/data/views.py`
- `tests/data/test_views.py`

**Implement:**

- `PartnerDesignView`
- default bidirectional expansion:
  - design A conditioned on B
  - design B conditioned on A
- construct:
  - `designed_residue_mask`
  - `fixed_residue_mask`
  - `mask_for_loss`
  - interface-view metadata

**Acceptance:**

- no training path assumes `designed_residue_mask = all True`
- every eligible example yields two views by default

### 2.4 — Mixed Dataset

Serve mixable partner-design views from multiple sources.

**Files:**

- `src/teddympnn/data/dataset.py`
- `tests/data/test_dataset.py`

**Requirements:**

- consume adapter manifests and cached normalized examples
- emit `PartnerDesignView`
- support configurable source mixing weights
- keep source provenance available for evaluation and ablations

**Acceptance:**

- dataset mixing ratios are approximately correct
- views contain designed/fixed masks with the intended semantics

### 2.5 — Token-Budget Batch Sampler

Move batch membership logic out of the collator.

**Files:**

- `src/teddympnn/data/sampler.py`
- `tests/data/test_sampler.py`

**Implement:**

- `TokenBudgetBatchSampler`
- group indices under a residue/token budget
- support model-family-specific budgets

**Acceptance:**

- batch grouping is implementable with standard `DataLoader` semantics
- no collator is responsible for fetching extra dataset items

### 2.6 — Padding Collator And Batch Builders

Pad and convert selected views into model-family-specific batches.

**Files:**

- `src/teddympnn/data/collator.py`
- `tests/data/test_collator.py`

**Implement:**

- padding collator only
- ProteinMPNN batch builder:
  - derive `X` from `xyz_37`
- LigandMPNN batch builder:
  - derive `X` from `xyz_37`
  - pass `xyz_37`, `xyz_37_mask`
  - build `Y / Y_m / Y_t` from real hetero atoms plus fixed-partner side-chain
    atoms

**Acceptance:**

- `X` always means backbone-only coordinates
- ProteinMPNN and LigandMPNN receive the documented batch contracts

## Phase 3: Training Stack

### 3.1 — Masked-Mean Label-Smoothed NLL

Implement the primary training loss.

**Files:**

- `src/teddympnn/training/loss.py`
- `tests/training/test_loss.py`

**Requirements:**

- label smoothing over the current vocabulary
- masked mean over `mask_for_loss`
- numerator/denominator form so DDP can reduce them globally before division

**Acceptance:**

- no fixed normalization constant
- loss scale is invariant to padding and token-budget composition

### 3.2 — Learning Rate Scheduler

Implement Noam scheduling.

**Files:**

- `src/teddympnn/training/scheduler.py`
- `tests/training/test_scheduler.py`

**Acceptance:**

- warmup and decay match the configured formula

### 3.3 — Trainer

Implement the training loop around the native checkpoint bundle.

**Files:**

- `src/teddympnn/training/trainer.py`
- `tests/training/test_trainer.py`

**Requirements:**

- single-device and DDP execution
- AMP support
- native bundle save/load for resume
- validation on recovery metrics

**Acceptance:**

- save/resume uses native bundle I/O
- DDP loss reduction uses global numerator/denominator semantics

### 3.4 — Configuration

Define configuration models for the new contracts.

**Files:**

- `src/teddympnn/config.py`
- `tests/test_config.py`

**Requirements:**

- explicit model-family config
- source-mixing config
- token-budget sampler config
- checkpoint export config

**Acceptance:**

- defaults distinguish ProteinMPNN and LigandMPNN where needed
- no config field implies all-residue redesign as the default

### 3.5 — CLI Integration

Expose training and checkpoint workflows.

**Files:**

- `src/teddympnn/cli.py`
- `tests/test_cli.py`

**Representative commands:**

- `teddympnn train --config PATH`
- `teddympnn evaluate recovery --checkpoint PATH --data PATH`
- `teddympnn evaluate ddg --checkpoint PATH --skempi PATH`
- `teddympnn checkpoints export-foundry --checkpoint PATH --output PATH`
- `teddympnn checkpoints import-legacy --model MODEL --input PATH --output PATH`

### 3.6 — Validation Gate: End-To-End Fine-Tuning Smoke Test

Run a small real-data fine-tuning smoke test for each model family.

**Files:**

- `tests/training/test_e2e_training.py`

**Acceptance:**

- loss decreases
- checkpoints resume correctly from native bundles
- Foundry export from a trained checkpoint succeeds

## Phase 4: Evaluation

### 4.1 — Sequence Recovery

**Files:**

- `src/teddympnn/evaluation/sequence_recovery.py`
- `tests/evaluation/test_sequence_recovery.py`

**Requirements:**

- overall recovery on designed residues
- interface-only recovery on designed interface residues
- metrics stratified by source

### 4.2 — ddG Scoring

Implement ddG prediction as a downstream scoring application built on the
sequence model.

**Files:**

- `src/teddympnn/evaluation/binding_affinity.py`
- `tests/evaluation/test_binding_affinity.py`

**Requirements:**

- complex and monomer scoring path
- mutation application
- Monte Carlo averaging and shared-randomness variance reduction

### 4.3 — SKEMPI Benchmark

**Files:**

- `src/teddympnn/evaluation/skempi.py`
- `tests/evaluation/test_skempi.py`

## Phase 5: Scale And Benchmark

### 5.1 — Full Data Preparation

- prepare teddymer, NVIDIA complexes, and optional PDB supplemental data
- cache normalized examples and partner-design views

### 5.2 — Full Fine-Tuning Runs

- run main ProteinMPNN and LigandMPNN fine-tuning jobs
- include dataset-mixing ablations

### 5.3 — Benchmarking

- compare against unfine-tuned ProteinMPNN and LigandMPNN
- report recovery and ddG benchmark results

### 5.4 — Release

- update README
- ship configs and examples
- publish compatibility guidance for Foundry export/import

## Dependency Summary

Core:

- `torch>=2.2`
- `pydantic>=2.0`
- `typer>=0.9`
- `rich>=13.0`
- `numpy>=1.24`
- `pandas>=2.0`
- `pyyaml>=6.0`
- `biopython>=1.80`
- `aiohttp>=3.9`
- `zstandard>=0.22`

Optional:

- `wandb>=0.15`

## Task Dependencies

```text
Phase 1:
  1.1 -> 1.2, 1.3
  1.2 + 1.3 -> 1.4, 1.5
  1.1 + 1.4 + 1.5 -> 1.6
  1.4 + 1.5 + 1.6 -> 1.7

Phase 2:
  2.1 -> 2.2, 2.3
  2.2 + 2.3 -> 2.4
  2.4 -> 2.5, 2.6

Phase 3:
  2.5 + 2.6 + 1.7 -> 3.1, 3.2, 3.3
  3.1 + 3.2 -> 3.3
  3.3 -> 3.4, 3.5, 3.6

Phase 4:
  3.6 -> 4.1, 4.2
  4.2 -> 4.3

Phase 5:
  2.x + 3.6 -> 5.1, 5.2
  4.x + 5.2 -> 5.3
  5.3 -> 5.4
```
