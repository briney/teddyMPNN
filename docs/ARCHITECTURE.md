# teddyMPNN Architecture

teddyMPNN is a standalone package for fine-tuning ProteinMPNN and LigandMPNN on
protein-protein interaction data. v1 is centered on interface-aware sequence
design. Relative binding affinity prediction is a downstream evaluation use case
built on the same scoring model, not a co-equal training objective.

## Goals

1. Improve interface sequence recovery on protein-protein complexes relative to
   unfine-tuned ProteinMPNN and LigandMPNN.
2. Support binder-style redesign by treating one partner group as designable and
   the other as fixed context.
3. Preserve checkpoint portability:
   - initialize from original ProteinMPNN and LigandMPNN checkpoints
   - export fine-tuned checkpoints that Foundry can strict-load
4. Keep the package operationally standalone:
   - no Foundry runtime dependency
   - no requirement to mirror Foundry's broader data or inference stack

## Non-Goals For v1

- Reproducing Foundry's runtime API surface exactly
- Introducing a new protein-partner encoder for LigandMPNN
- Joint full-complex redesign as the default training mode
- Supervised ddG fine-tuning

## Design Principles

- **Standalone runtime**: Use Foundry as a reference implementation, not as a
  dependency.
- **Checkpoint parity over runtime parity**: Match parameter names, tensor
  layouts, token ordering, and feature ordering where portability depends on
  them.
- **Native-first artifacts**: Use a teddyMPNN-native checkpoint bundle as the
  canonical training artifact, with explicit export adapters for Foundry-current
  and original legacy formats.
- **Separate adapters, shared contracts**: Keep source-specific data ingestion
  separate, but normalize all datasets into one internal structure contract.
- **Bidirectional partner design by default**: Every eligible complex produces
  two training views by swapping design and context partner groups.

## System Overview

```text
raw structures / source metadata
    -> dataset-specific adapters
    -> InterfaceExample
    -> PartnerDesignView expansion
    -> model-family-specific batch builder
    -> ProteinMPNN or LigandMPNN core
    -> native CheckpointBundle
    -> explicit export adapters (Foundry-current, legacy original)
```

## Data Contracts

### InterfaceExample

`InterfaceExample` is the normalized parsed-structure record shared by all data
adapters. It is structure-level, not training-view-level.

Required fields:

| Field | Shape | Purpose |
| --- | --- | --- |
| `example_id` | scalar | Stable identifier |
| `source` | scalar | `teddymer`, `nvidia_complexes`, or `pdb_complexes` |
| `xyz_37` | `(L, 37, 3)` | Per-residue atom coordinates in one canonical 37-atom order |
| `xyz_37_mask` | `(L, 37)` | Atom-resolved mask |
| `S` | `(L,)` | Residue tokens in current token order |
| `R_idx` | `(L,)` | Residue indices |
| `chain_ids` | `(L,)` | Original chain identifiers |
| `chain_labels` | `(L,)` | Integer chain labels |
| `residue_mask` | `(L,)` | Valid residue mask |
| `partner_group` | `(L,)` | Integer group label indicating which partner each residue belongs to |
| `interface_residue_mask` | `(L,)` | Residues participating in the interface |
| `hetero_Y` | `(N, 3)` | Real non-protein atom coordinates |
| `hetero_Y_m` | `(N,)` | Real non-protein atom mask |
| `hetero_Y_t` | `(N,)` | Real non-protein atom types |
| `metadata` | mapping | Quality scores, provenance, split labels, source-specific annotations |

Notes:

- `xyz_37` is the parsed all-atom representation. It is not fed unchanged to
  every model.
- v1 normalizes training views to **two partner groups**: a design group and a
  context group. Each group may contain one or more chains.
- Dataset provenance is preserved through `metadata`; it is not erased during
  normalization.

### PartnerDesignView

`PartnerDesignView` is derived from `InterfaceExample` and is the unit consumed
by training and most evaluation code.

Required fields:

| Field | Shape | Purpose |
| --- | --- | --- |
| `example` | object | Back-reference to the parent `InterfaceExample` |
| `design_partner_group` | scalar | Which partner group is being designed |
| `context_partner_group` | scalar | Which partner group is fixed context |
| `designed_residue_mask` | `(L,)` | Residues predicted by the model |
| `fixed_residue_mask` | `(L,)` | Complementary context residues |
| `mask_for_loss` | `(L,)` | Valid residues contributing to the loss |
| `interface_residue_mask` | `(L,)` | Interface subset for evaluation and optional weighting |
| `view_metadata` | mapping | Direction (`A<-B` vs `B<-A`), source, crop info, etc. |

Default expansion policy:

- Every eligible two-partner complex yields two views:
  - design partner 0 conditioned on partner 1
  - design partner 1 conditioned on partner 0
- This is the default for both ProteinMPNN and LigandMPNN fine-tuning.

## Model Batch Contracts

`X` has one meaning throughout the project:

- `X`: backbone-only coordinates with shape `(B, L, 4, 3)` in the order
  `N, CA, C, O`

Parsed all-atom data remains available separately as `xyz_37` and
`xyz_37_mask`.

### ProteinMPNN Batch

Shared fields:

| Field | Shape |
| --- | --- |
| `X` | `(B, L, 4, 3)` |
| `S` | `(B, L)` |
| `R_idx` | `(B, L)` |
| `chain_labels` | `(B, L)` |
| `residue_mask` | `(B, L)` |
| `designed_residue_mask` | `(B, L)` |
| `fixed_residue_mask` | `(B, L)` |
| `mask_for_loss` | `(B, L)` |

ProteinMPNN never consumes the full 37-atom tensor directly.

### LigandMPNN Batch

LigandMPNN uses the same shared fields plus:

| Field | Shape |
| --- | --- |
| `xyz_37` | `(B, L, 37, 3)` |
| `xyz_37_mask` | `(B, L, 37)` |
| `Y` | `(B, N, 3)` |
| `Y_m` | `(B, N)` |
| `Y_t` | `(B, N)` |
| `atomize_side_chains` | scalar or `(B,)` |
| `hide_side_chain_mask` | `(B, L)` |

For PPI fine-tuning, `Y / Y_m / Y_t` are built from:

- real non-protein atoms already present in the structure
- fixed-partner side-chain atoms exposed through the existing LigandMPNN context
  pathway

Designed-partner side chains are not exposed as context during training.

## Model Families

### ProteinMPNN Core

The ProteinMPNN core is a checkpoint-compatible reimplementation of the
reference architecture:

- same high-level module tree where state-dict portability depends on it
- same hidden size, encoder depth, decoder depth, and edge-feature layout as
  the reference weights
- same token ordering and output projection ordering as the current Foundry
  MPNN implementation

Backbone featurization:

- Use atoms `N, CA, C, O` plus a virtual `CB` computed from backbone geometry.
- Build the k-nearest-neighbor graph from `CA-CA` distances.
- Use 25 ordered backbone/virtual-atom pairs for radial basis features.

Positional encoding:

- Clip residue offsets to `[-32, 32]`
- One-hot encode `65` offset bins plus `1` inter-chain bucket
- Total classes: `66`
- Inter-chain bucket index: `65`

### LigandMPNN Core

LigandMPNN preserves the existing ligand-context architecture rather than
introducing a new partner encoder.

Implications:

- The backbone encoder remains checkpoint-compatible with the reference design.
- Protein-partner atomic context is injected through the existing
  protein-to-context and context-subgraph pathway.
- This keeps pretrained-weight import and Foundry export tractable.

LigandMPNN context for PPI fine-tuning:

- Fixed-partner side-chain atoms are surfaced as context atoms.
- Real hetero atoms remain in the same context pool.
- The model still uses the reference-style context encoder rather than a new
  PPI-specific module.

## Weight Compatibility And Checkpoints

### Compatibility Boundary

Portability depends on:

- parameter and buffer names
- token order
- radial-basis atom-pair order
- atom-type vocabulary conventions
- positional-encoding configuration

The architecture therefore mirrors the reference implementation only where
those boundaries matter. teddyMPNN does **not** mirror Foundry's full runtime,
data pipeline, or CLI.

### Canonical Artifact: `CheckpointBundle`

teddyMPNN stores training state in a native bundle:

```python
{
    "format_version": "teddympnn.v1",
    "model_family": "protein_mpnn" | "ligand_mpnn",
    "model_config": {...},
    "training_config": {...},
    "state_dict": {...},
    "optimizer_state": {...} | None,
    "scheduler_state": {...} | None,
    "step": int,
    "metrics": {...},
    "compatibility": {
        "token_order": [...],
        "legacy_token_order": [...],
        "rbf_pair_order": [...],
        "atom_type_vocab_size": int,
        "positional_max_relative_feature": 32,
        "positional_num_classes": 66,
    },
}
```

This is the canonical artifact for:

- checkpoint resume
- provenance
- reproducibility
- future format evolution

### Import And Export Adapters

Supported compatibility paths:

- import current Foundry-style checkpoints
- import original legacy ProteinMPNN and LigandMPNN checkpoints
- export Foundry-current strict-loadable checkpoints
- export legacy-format checkpoints when feasible

Foundry compatibility remains a first-class requirement, but Foundry-format
checkpoints are **export targets**, not the internal training artifact.

## Dataset Adapters And Mixing

### teddymer Adapter

Responsibilities:

- parse source metadata and domain boundaries
- assemble normalized two-partner examples
- preserve source-specific quality metadata and cluster annotations

### NVIDIA Complex Adapter

Responsibilities:

- filter metadata using confidence thresholds
- map passing examples to chunk archives
- normalize extracted structures into `InterfaceExample`

Practical note:

- The metadata filter may still touch a large fraction of archives.
- Chunk-coverage profiling is part of data preparation, not an afterthought.

### PDB Complex Adapter

Responsibilities:

- curate experimental multi-chain complexes
- normalize them into the same contract as predicted-structure datasets

### Mixing Strategy

- Adapters remain source-specific.
- Training consumes a configurable mixture of normalized examples.
- Dataset provenance is preserved for sampling, evaluation, and ablations.

## Training Pipeline

### Default Objective

The default fine-tuning objective is **bidirectional partner design**.

For each `InterfaceExample`:

1. Build a view that designs partner group A while fixing partner group B.
2. Build a second view that designs partner group B while fixing partner group A.

This objective applies to both model families.

### Loss

Primary loss: label-smoothed negative log-likelihood over designed positions.

```text
numerator   = sum(mask_for_loss * per_position_nll)
denominator = sum(mask_for_loss)
loss        = numerator / max(denominator, 1)
```

Distributed training uses global numerator and denominator reduction before the
final division so the loss is invariant to padding, token budget, and replica
layout.

### Optimization

- Optimizer: Adam with reference-style hyperparameters
- Scheduler: Noam warmup + inverse-square-root decay
- Structure noise defaults follow the upstream pretrained families
- Full-model fine-tuning is the default; no LoRA or adapter heads in v1

### Batch Construction

Batching is split into two layers:

- `TokenBudgetBatchSampler`: groups example indices under a residue/token budget
- padding collator: stacks and pads already chosen batches

The collator does **not** decide batch membership.

## Evaluation

### Sequence Recovery

Primary evaluation:

- overall recovery on designed residues
- interface-only recovery on designed interface residues
- metrics stratified by source and structure properties

### ddG As Downstream Evaluation

ddG prediction is a downstream scoring application built on the sequence model.
It remains part of the evaluation stack, but it is not a core v1 training
objective.

## Package Structure

```text
src/teddympnn/
├── cli.py
├── config.py
├── models/
│   ├── protein_mpnn.py
│   ├── ligand_mpnn.py
│   ├── tokens.py
│   └── layers/
├── data/
│   ├── contracts.py
│   ├── features.py
│   ├── views.py
│   ├── dataset.py
│   ├── sampler.py
│   ├── collator.py
│   ├── teddymer.py
│   ├── nvidia_complexes.py
│   └── pdb_complexes.py
├── training/
│   ├── loss.py
│   ├── scheduler.py
│   └── trainer.py
├── evaluation/
│   ├── sequence_recovery.py
│   ├── binding_affinity.py
│   └── skempi.py
└── weights/
    ├── io.py
    ├── foundry.py
    └── legacy.py
```

## CLI Surface

Representative commands:

```bash
teddympnn train --config configs/protein_mpnn_ppi.yaml
teddympnn evaluate recovery --checkpoint outputs/run.ckpt --data data/test/
teddympnn evaluate ddg --checkpoint outputs/run.ckpt --skempi data/skempi/
teddympnn checkpoints export-foundry --checkpoint outputs/run.ckpt --output foundry.pt
teddympnn checkpoints import-legacy --model protein_mpnn --input proteinmpnn_v_48_020.pt --output init.ckpt
```

## Dependencies

Core runtime:

- `torch`
- `pydantic`
- `typer`
- `rich`
- `numpy`
- `pandas`
- `pyyaml`
- `biopython`
- `aiohttp`
- `zstandard`

Optional:

- `wandb`
