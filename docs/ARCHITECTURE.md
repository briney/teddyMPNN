# teddyMPNN Architecture

> For the project vision, goals, and non-goals, see [VISION.md](VISION.md).

teddyMPNN fine-tunes ProteinMPNN on the teddymer dataset to improve interface
sequence design. This document covers the current implementation: model,
training pipeline, and evaluation.

---

## Model: ProteinMPNN

~1.66M parameters. Initialized from the bundled `proteinmpnn_v_48_020` base
weights. Module hierarchy mirrors Foundry's exactly for strict `state_dict`
loading.

```
ProteinMPNN (nn.Module)
├── graph_featurization_module: ProteinFeatures
│   ├── positional_embedding: PositionalEncodings
│   ├── edge_embedding: Linear(num_edge_input_features → num_edge_output_features)
│   └── edge_norm: LayerNorm(num_edge_output_features)
├── W_e: Linear(num_edge_features → hidden_dim)
├── W_s: Embedding(vocab_size, hidden_dim)
├── encoder_layers: ModuleList[EncLayer × 3]
├── decoder_layers: ModuleList[DecLayer × 3]
└── W_out: Linear(hidden_dim → vocab_size)
```

**Key hyperparameters:**

| Parameter | Value |
|-----------|-------|
| `hidden_dim` | 128 |
| `num_encoder_layers` | 3 |
| `num_decoder_layers` | 3 |
| `num_neighbors` (k) | 48 |
| `vocab_size` | 21 (20 AA + unknown) |
| `num_rbf` | 16 |
| `rbf_range` | 2.0 – 22.0 Å |
| `num_positional_embeddings` | 16 |
| `aggregation_scale` | 30 |
| `ffn_expansion` | 4x |
| `dropout` | 0.1 |

**Backbone representation:** N, CA, C, O, and virtual CB (computed from
N/CA/C). k-NN graph from CA-CA distances.

**Edge features:** 25 atom-pair × 16 RBF = 400 dims + 16-dim positional
encoding → projected to 128.

**Relative positional encoding:** 66 classes — 65 intra-chain offsets (−32…32)
+ one inter-chain bucket at index 65. Must stay fixed for checkpoint
compatibility.

---

## Training Pipeline

```
teddymer dimers
  → featurize (backbone k-NN graph + interface mask)
  → ProteinMPNN (autoregressive decoder)
  → interface-weighted cross-entropy
  → checkpoint
```

Each dimer is expanded into two partner-design views: design chain A
conditioning on B, and design chain B conditioning on A.

### Loss

Label-smoothed cross-entropy (ε = 0.1) over designed positions, with
per-residue weights upweighting interface residues (inter-chain neighbors
within 8 Å CB-CB). The `interface_weight` config knob (default 1.0) scales
the interface loss; 1.0 = standard cross-entropy, values > 1.0 increase
interface emphasis.

### Optimizer / scheduler

- Adam (β₁=0.9, β₂=0.98, ε=1e-9)
- Noam scheduler: `lr = factor × d_model^(−0.5) × min(step^(−0.5), step × warmup^(−1.5))`
  - Defaults: `factor=2`, `warmup_steps=4000`

### Checkpoints

Each checkpoint saves a teddyMPNN-native bundle:

```python
{
    "format_version": "teddympnn.v1",
    "state_dict": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "step": global_step,
    "config": config.model_dump(),
    "metrics": {...},
}
```

---

## Data

### Training: teddymer

~510K quality-filtered cluster representatives of synthetic protein-protein
dimers (AFDB domains split at CATH boundaries). Filtering criteria:
interface pLDDT > 70, ipAE < 10, interface length > 10 residues.

Pipeline: download teddymer archive → parse metadata + source indices →
fetch TED-domain PDBs → relabel chains A/B → write dimers + manifests.

### Evaluation: PDB experimental complexes

Real experimental multi-chain structures (X-ray / cryo-EM, resolution < 3.5 Å,
≥2 protein chains with interface contacts). Used for generalization eval only.

### Evaluation: SKEMPI v2.0

7,085 binding ΔΔG measurements across 345 complexes. Used for ddG correlation
evaluation.

---

## Evaluation

### Interface sequence recovery

- Define interface as residues with ≥1 neighbor within `interface_cutoff` (8 Å
  CB-CB) of the partner chain.
- Score: argmax accuracy at interface positions vs. ground truth.
- Reported on held-out teddymer split (in-distribution) and PDB complexes
  (generalization).

### Binding affinity prediction (ΔΔG)

Thermodynamic decomposition (StaB-ddG / BA-DDG):

```
ΔΔG(wt → mut) ≈ [log p(mut | X_AB) − log p(mut | X_A) − log p(mut | X_B)]
              − [log p(wt  | X_AB) − log p(wt  | X_A) − log p(wt  | X_B)]
```

6 forward passes per mutation. Antithetic variates + Monte Carlo averaging
(default 20 samples) for variance reduction.

---

## Weight I/O

**Loading:** The Foundry base checkpoint loads with `strict=True`. Legacy
checkpoint format (`model_state_dict` key, 1-letter token order, legacy RBF
pair order) is supported via key transformations in `weights/foundry.py`.

**Saving:** Plain teddyMPNN native bundles — no re-export to Foundry or legacy
formats.

---

## CLI

```bash
# Train (Hydra-style overrides)
python -m teddympnn train [--config configs/train.yaml] [--resume CKPT] [OVERRIDES...]

# Score a structure
python -m teddympnn score --checkpoint CKPT --pdb STRUCTURE.pdb --chains A [--num-samples N]

# Evaluate interface sequence recovery
python -m teddympnn evaluate recovery --checkpoint CKPT --data MANIFEST.tsv

# Evaluate binding affinity (SKEMPI)
python -m teddympnn evaluate ddg --checkpoint CKPT --skempi DATA_DIR [--num-samples N]
```

---

## Package Structure

```
src/teddympnn/
├── cli.py                       # Typer CLI (train, score, evaluate, download)
├── config.py                    # Pydantic config models
├── models/
│   ├── protein_mpnn.py          # ProteinMPNN nn.Module
│   ├── tokens.py                # Amino-acid token vocabulary
│   └── layers/                  # EncLayer, DecLayer, ProteinFeatures, etc.
├── data/
│   ├── dataset.py               # PPIDataset
│   ├── teddymer.py              # Teddymer download + preprocessing
│   ├── pdb_complexes.py         # PDB experimental complexes
│   ├── features.py              # PDB/mmCIF → feature tensors
│   ├── splits.py                # Train/val/test split utilities
│   ├── sampler.py               # Token-budget batch sampler
│   └── collator.py              # Padding collator
├── training/
│   ├── trainer.py               # Training loop
│   ├── loss.py                  # Interface-weighted cross-entropy
│   └── scheduler.py             # NoamScheduler
├── evaluation/
│   ├── sequence_recovery.py     # Interface recovery metrics
│   ├── binding_affinity.py      # ΔΔG prediction
│   └── skempi.py                # SKEMPI v2.0 utilities
└── weights/
    ├── io.py                    # Checkpoint bundle I/O
    ├── foundry.py               # Foundry checkpoint loading
    └── legacy.py                # Legacy dauparas/IPD checkpoint loading
```
