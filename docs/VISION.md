# teddyMPNN — Vision

> **North star:** Fine-tune ProteinMPNN on a large multimer dataset so it
> designs better residues at protein–protein interfaces than stock ProteinMPNN.
> Keep it simple. One purpose, one training dataset, no extra machinery.

This document is the reference for an in-place simplification of the current
codebase, which has grown far beyond what a focused fine-tune needs. When in
doubt, delete.

---

## The idea

ProteinMPNN ([Dauparas et al., 2022](https://www.science.org/doi/10.1126/science.add2187))
decodes each residue conditioned on the full backbone structure of a complex,
so it already has implicit awareness of interfaces. We sharpen that awareness by
fine-tuning the public base weights on
[teddymer](https://teddymer.steineggerlab.workers.dev/), a large dataset of
predicted protein dimers, with the loss weighted toward interface residues.

The result is a drop-in ProteinMPNN replacement that picks better interface
residues for tasks like affinity maturation and interface redesign.

---

## Scope

**In scope**

- **Interface sequence design** — the core capability. Fine-tune ProteinMPNN on
  teddymer; measure interface sequence recovery.
- **ddG / affinity prediction** — score binding-affinity changes on SKEMPI v2.0
  as a downstream check that better interface modeling transfers.

**Out of scope (non-goals)**

- De novo binder design.
- Multi-model benchmarking harness.
- LigandMPNN and any ligand/nucleotide context — teddymer is protein-only
  dimers, so this path is dead weight.
- Bidirectional Foundry / legacy checkpoint export.
- Multiple training datasets or a general-purpose data framework.

---

## Guiding principles

1. **One purpose.** Every module serves the interface fine-tune. If it doesn't,
   it goes.
2. **Extend, don't abstract.** The validated model, the interface-mask helper,
   and the masked loss already exist. Build on them; don't add layers of
   indirection.
3. **One training source.** teddymer in, fine-tuned weights out. No data-source
   plugin system.
4. **Simple checkpoints.** Load the public base weights once to initialize; save
   plain teddyMPNN checkpoints (model + optimizer + scheduler + step). No
   round-tripping to other formats.
5. **Delete first.** Removing code is the primary work of this refactor.

---

## Architecture

The model is kept as-is — it passes Foundry-equivalence tests and needs no
changes. Everything around it is trimmed.

**Model.** ProteinMPNN: 3 encoder + 3 decoder message-passing layers, 128-dim
hidden state, k=48 neighbors, 21-token vocabulary (~1.66M parameters).
Initialized from the bundled `proteinmpnn_v_48_020` base weights.

**Pipeline.**

```
teddymer dimers
  → featurize (backbone k-NN graph + interface mask)
  → ProteinMPNN
  → interface-weighted cross-entropy
  → checkpoint
```

**Training objective.** Standard autoregressive, label-smoothed cross-entropy,
with per-residue weights upweighting interface residues (those with inter-chain
neighbors, via the existing `identify_interface_residues` helper). Controlled by
a single `interface_weight` config knob; `1.0` reduces to plain ProteinMPNN
training.

**Evaluation.**

- *Interface sequence recovery* on a held-out teddymer split (in-distribution)
  and on real PDB experimental complexes (generalization).
- *ddG correlation* on SKEMPI v2.0.

**Configuration.** OmegaConf with Hydra-style CLI overrides (e.g.
`train.interface_weight=3.0`). No bespoke flag-per-knob CLI.

**CLI.** Three commands: `train`, `score`, `evaluate {recovery, ddg}`.

---

## Refactor plan

Surgical, in-place trim on this branch. Disposition of the current tree:

| Area | Keep | Simplify | Delete |
|------|------|----------|--------|
| `models/` | `protein_mpnn`, `tokens`, `layers/*` | — | `ligand_mpnn` |
| `data/` | `pdb_complexes`, `collator`, `features` | `teddymer`, `dataset`, `splits`, `sampler` | `nvidia_complexes` |
| `evaluation/` | `sequence_recovery`, `skempi`, `_batch` | `binding_affinity` (fold into ddG path) | `benchmark` |
| `training/` | `scheduler` | `trainer`, extend `loss` for interface weighting | — |
| `weights/` | `pretrained/`, `io` | `legacy`, `foundry` (load-only) | export-to-Foundry/legacy paths |
| `config` / `cli` | `config` | `cli` (3 commands) | benchmark command |
| `docs/` | `VISION.md` | shrink `ARCHITECTURE.md` | `WORKPLAN.md`, `TECHNICAL_ANALYSIS.md` |
| `scripts/` | `prepare_data`, `launch_training`, `download_pretrained_weights` | — | `run_benchmark` |

Tests follow their modules: delete tests for deleted code, keep the
Foundry-equivalence test for the model.

---

## Success criteria

The refactor and fine-tune succeed when:

1. teddyMPNN **beats base ProteinMPNN on interface sequence recovery** (held-out
   teddymer split and PDB complexes).
2. It **does not regress** non-interface sequence recovery.
3. It **≥ matches base ProteinMPNN on SKEMPI ddG correlation**.
4. The codebase is materially smaller and a new reader can trace
   teddymer → fine-tuned weights end to end without a map.
