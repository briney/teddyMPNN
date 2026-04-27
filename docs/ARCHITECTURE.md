# teddyMPNN Architecture

Fine-tuning ProteinMPNN and LigandMPNN on protein-protein interaction data for
interface sequence design and relative binding affinity prediction.

## Goals

1. **Interface sequence design** — Improve sequence recovery at protein-protein
   interfaces beyond what generic ProteinMPNN/LigandMPNN achieve.
2. **De novo binder design** — Fix a target chain, redesign a binder chain, with
   interface-aware scoring.
3. **Relative binding affinity (ddG) prediction** — Use inverse folding
   log-likelihoods as a proxy for binding free energy changes upon mutation,
   following the StaB-ddG / BA-DDG thermodynamic decomposition.

## Models

We reimplement ProteinMPNN and LigandMPNN as standalone PyTorch modules,
preserving the checkpoint-relevant module hierarchy, attribute names, tensor
shapes, token ordering, and feature ordering needed for transferable
`state_dict` loading. This enables:

- Loading pretrained weights from Foundry checkpoints without conversion
- Exporting fine-tuned checkpoints that Foundry users can load directly

### ProteinMPNN (~1.66M parameters)

> **Pending Foundry validation**: the current implementation produces 1,660,485
> parameters; a 3,504-parameter divergence from the originally cited Foundry
> figure (1,656,981). The discrepancy is not yet diagnosed and must be resolved
> by loading reference Foundry weights with `strict=True` before Phase 5.


```
ProteinMPNN (nn.Module)
├── graph_featurization_module: ProteinFeatures
│   ├── positional_embedding: PositionalEncodings
│   │   └── embed_positional_features: Linear(num_embeddings → num_positional_embeddings)
│   ├── edge_embedding: Linear(num_edge_input_features → num_edge_output_features)
│   └── edge_norm: LayerNorm(num_edge_output_features)
├── W_e: Linear(num_edge_features → hidden_dim)
├── W_s: Embedding(vocab_size, hidden_dim)
├── encoder_layers: ModuleList[EncLayer × 3]
│   └── EncLayer
│       ├── W1, W2, W3: Linear        # node message MLP
│       ├── W11, W12, W13: Linear      # edge update MLP
│       ├── norm1, norm2, norm3: LayerNorm
│       ├── dropout1, dropout2, dropout3: Dropout
│       └── dense: PositionWiseFeedForward(hidden_dim, hidden_dim * 4)
├── decoder_layers: ModuleList[DecLayer × 3]
│   └── DecLayer
│       ├── W1, W2, W3: Linear
│       ├── norm1, norm2: LayerNorm
│       ├── dropout1, dropout2: Dropout
│       └── dense: PositionWiseFeedForward(hidden_dim, hidden_dim * 4)
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
| `rbf_range` | 2.0 – 22.0 A |
| `num_positional_embeddings` | 16 |
| `aggregation_scale` | 30 |
| `ffn_expansion` | 4x |
| `dropout` | 0.1 |

**Backbone representation:** 5 atoms per residue — N, CA, C, O, and a virtual CB
computed geometrically from N/CA/C. The k-nearest neighbor graph is built from
CA-CA distances.

**Edge features:** 25 atom-pair RBF distance encodings (all pairs of
{N, CA, C, O, CB}) × 16 RBF kernels = 400 dims, concatenated with 16-dim
positional encodings, projected through Linear + LayerNorm to 128 dims.

**Node features:** Initialized as zeros; learned entirely through message passing.

**Message passing:** Encoder layers update both node and edge hidden states via
concatenation-based MLP messages with GELU activation, scaled sum aggregation
(÷30), residual connections, LayerNorm, and position-wise feedforward.

**Decoding:** Autoregressive with randomized decoding order. A causal mask
ensures each position only attends to previously decoded neighbors. The final
linear layer projects to 21 amino acid logits.

**Relative positional encoding:** For `max_relative_feature=32`, we one-hot
encode 66 classes total: 65 clipped intra-chain offsets (`-32..32`) plus one
inter-chain bucket at index 65. This fixes the input dimensionality of
`embed_positional_features` and must remain stable for strict checkpoint
compatibility.

### LigandMPNN (~2.62M parameters)

> **Pending Foundry validation**: the current implementation produces 2,618,501
> parameters; a 3,472-parameter divergence from the originally cited Foundry
> figure (2,621,973). Resolve as part of the same pre-Phase-5 validation step.


Extends ProteinMPNN with a protein-ligand context encoder that operates over
three graphs:

```
LigandMPNN (extends ProteinMPNN)
├── [all ProteinMPNN modules]
├── graph_featurization_module: ProteinFeaturesLigand  # replaces ProteinFeatures
│   ├── [all ProteinFeatures modules]
│   ├── embed_atom_type_features: Linear(atom_type_input → atom_type_output)
│   ├── node_embedding: Linear(node_input → node_output)
│   ├── node_norm: LayerNorm
│   ├── ligand_subgraph_node_embedding: Linear(atom_type_input → node_output)
│   ├── ligand_subgraph_node_norm: LayerNorm
│   ├── ligand_subgraph_edge_embedding: Linear(num_rbf → node_output)
│   ├── ligand_subgraph_edge_norm: LayerNorm
│   └── [buffers: side_chain_atom_types, periodic_table_groups, periodic_table_periods]
├── W_protein_to_ligand_edges_embed: Linear(node_features → hidden_dim)
├── W_protein_encoding_embed: Linear(hidden_dim → hidden_dim)
├── W_ligand_nodes_embed: Linear(hidden_dim → hidden_dim)
├── W_ligand_edges_embed: Linear(hidden_dim → hidden_dim)
├── W_final_context_embed: Linear(hidden_dim → hidden_dim, bias=False)
├── final_context_norm: LayerNorm
├── protein_ligand_context_encoder_layers: ModuleList[DecLayer × 2]
└── ligand_context_encoder_layers: ModuleList[DecLayer × 2]
```

**Three graphs:**

1. **Protein backbone graph** — Same as ProteinMPNN but with k=32 (reduced from 48).

2. **Protein-ligand graph** — For each residue, selects the 25 closest non-protein
   atoms by virtual CB distance. Edge features: backbone-to-context RBF (80 dims)
   + atom type embeddings (64 dims) + angle features (4 dims) = 148 dims →
   projected to 128.

3. **Intraligand graph** — Ligand atoms as nodes with element type + periodic table
   group/period embeddings. Edges encode inter-atom RBF distances.

**Context encoder:** 2 decoder-style layers process the intraligand graph, then
2 layers process the protein-ligand graph. The resulting context representation
is projected and added to protein node embeddings before the standard decoder.

**Why LigandMPNN matters for PPI:**

- **Side-chain atomization** — LigandMPNN can treat partner chain side-chain atoms
  as explicit context, providing richer interface information than backbone-only.
- **Glycan handling** — Viral glycoproteins carry glycan modifications that
  ProteinMPNN cannot represent. LigandMPNN's unified non-protein atom framework
  encodes glycans (and other post-translational modifications) natively via
  element type embeddings.
- **Metals and cofactors** — Interface-proximal metal ions and cofactors are
  encoded through the same ligand context mechanism.

## Weight Compatibility

### Loading pretrained weights

Our module hierarchy mirrors Foundry's exactly, so loading is:

```python
checkpoint = torch.load(path, map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model"])
```

We also support Foundry's legacy checkpoint format (`model_state_dict` key)
with the same key transformations Foundry implements:

- **Token reordering:** Legacy uses 1-letter alphabetical (A, C, D, E, ...);
  current uses 3-letter alphabetical (Ala, Arg, Asn, Asp, ...). Affects `W_s`
  and `W_out`.
- **RBF pair reordering:** Legacy groups by same-atom pairs first; current uses
  outer-product order. Affects `edge_embedding`.
- **Atom type vocabulary:** Legacy has 120 types; current has 119 (drops unused
  index). Affects LigandMPNN embeddings.

### Exporting fine-tuned weights

The canonical training artifact is a teddyMPNN-native checkpoint bundle that
stores the model state alongside the compatibility metadata needed to safely
re-export into other ecosystems:

```python
{
    "format_version": "teddympnn.v1",
    "model_family": "protein_mpnn" | "ligand_mpnn",
    "state_dict": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "step": global_step,
    "config": config.model_dump(),
    "metrics": {...},
    "compatibility": {
        "token_order": "foundry_current",
        "rbf_pair_order": "foundry_current",
        "atom_type_vocabulary": "foundry_current",
        "positional_encoding_classes": 66,
    },
}
```

Foundry-compatible checkpoints are exported explicitly from this bundle, and a
`convert_to_legacy` utility is provided for users of the original
dauparas/ProteinMPNN and dauparas/LigandMPNN repos (reverse token/RBF/atom
reordering).

## Training Datasets

### 1. Teddymer — Synthetic dimers from AFDB domain pairs

**Source:** ~510K quality-filtered cluster representatives of synthetic
protein-protein dimers, constructed by splitting multi-domain AFDB monomers at
CATH domain boundaries.

**Filtering criteria (from original paper):**
- Interface pLDDT > 70
- Interface PAE (ipAE) < 10
- Interface length > 10 residues

**Data acquisition pipeline:**

```
┌──────────────────────────────────────────────────────────────┐
│  1. Download domain boundary table from Zenodo               │
│     (ted_100_324m.domain_summary.cath.globularity.taxid.tsv) │
├──────────────────────────────────────────────────────────────┤
│  2. Download teddymer cluster/metadata from                  │
│     teddymer.steineggerlab.workers.dev/foldseek/teddymer.tar │
├──────────────────────────────────────────────────────────────┤
│  3. Parse cluster.tsv + nonsingletonrep_metadata.tsv         │
│     → extract UniProt IDs and domain boundaries              │
├──────────────────────────────────────────────────────────────┤
│  4. Download full-chain PDBs from AFDB                       │
│     (~500K unique chains via HTTPS at ~35ms/req)             │
├──────────────────────────────────────────────────────────────┤
│  5. Chop domains locally using pdb-tools pdb_selres          │
│     (instantaneous per file)                                 │
├──────────────────────────────────────────────────────────────┤
│  6. Assemble dimers: relabel chains (A/B), concatenate,      │
│     write combined PDB                                       │
└──────────────────────────────────────────────────────────────┘
```

**Estimated cost:** ~500K AFDB downloads (~125 GB), ~1M domain chops, ~500K
dimer assemblies. With 50 concurrent workers: ~1 hour total.

**Why not the TED API directly:** The TED API generates domain PDBs on-the-fly
by downloading from AFDB and chopping — replicating that locally avoids
hammering both servers and is faster in aggregate.

### 2. NVIDIA/EMBL-EBI predicted complexes

**Source:** ~1.8M high-confidence predicted protein complexes (AlphaFold-Multimer)
from 4,777 proteomes, filtered from ~31M total predictions.

**Filtering criteria (from original paper):**
- ipSAEmin >= 0.6
- pLDDTavg >= 70
- Backbone clashes <= 10

**Data acquisition pipeline:**

```
┌────────────────────────────────────────────────────────────────┐
│  1. Download model_entity_metadata_mapping.csv (4.3 GB)        │
├────────────────────────────────────────────────────────────────┤
│  2. Filter rows by confidence thresholds                       │
│     → identify which chunk tarballs contain passing structures │
├────────────────────────────────────────────────────────────────┤
│  3. Download only the required chunk_NNNN.tar files            │
│     (each ~7.5 GB, zstd-compressed contents)                  │
├────────────────────────────────────────────────────────────────┤
│  4. Extract + decompress (zstd) passing structures only        │
│     → mmCIF files                                              │
├────────────────────────────────────────────────────────────────┤
│  5. Parse mmCIF → extract backbone + side-chain coordinates,   │
│     chain labels, confidence scores                            │
└────────────────────────────────────────────────────────────────┘
```

**Scale consideration:** Even after filtering to ~1.8M structures, the data
spans most of the 4,000 chunk tarballs (~30 TB total). A secondary filtering
step (e.g., ipSAEmin >= 0.8 for "very high confidence" only, yielding ~970K
structures) may be needed to keep download volume practical. We should profile
chunk coverage after the metadata filter to decide.

### 3. PDB experimental complexes (supplementary)

We also support loading experimental multi-chain structures from the PDB,
following Foundry's existing filters:

- Resolution < 3.5 A
- Method: X-ray diffraction or cryo-EM (no NMR)
- At least 2 protein chains with interface contacts

This provides high-quality ground truth to mix with the predicted-structure
datasets. The original teddymer paper used an 8:2 teddymer:PDB mixing ratio.

### Data mixing

Training data is sampled from a configurable mixture of datasets:

```yaml
# Example config
data:
  sources:
    - name: teddymer
      weight: 0.6
      path: /data/teddymer/dimers/
    - name: nvidia_complexes
      weight: 0.2
      path: /data/nvidia/filtered/
    - name: pdb_complexes
      weight: 0.2
      path: /data/pdb/complexes/
  token_budget: 10000     # max residues per batch
  max_residues: 6000      # max residues per structure
  min_interface_contacts: 4
```

### Training view generation

Each two-partner complex is expanded into two partner-design training views by
default:

- Design partner A while conditioning on partner B
- Design partner B while conditioning on partner A

At the batch level, `designed_residue_mask` marks exactly one partner group,
while `fixed_residue_mask` marks the complementary conditioning partner. This
keeps the training objective aligned with binder/interface design rather than
joint all-residue redesign of the full complex.

## Feature Computation

Both models consume the same core feature dictionary, with LigandMPNN adding
ligand-specific fields:

### Shared features (ProteinMPNN + LigandMPNN)

| Feature | Shape | Description |
|---------|-------|-------------|
| `xyz_37` | `(B, L, 37, 3)` | Parsed all-atom residue coordinates |
| `xyz_37_m` | `(B, L, 37)` | Parsed all-atom validity mask |
| `X` | `(B, L, 4, 3)` | Backbone-only coordinates (N, CA, C, O) derived from `xyz_37` |
| `X_m` | `(B, L, 4)` | Backbone-only atom validity mask |
| `S` | `(B, L)` | Amino acid token indices |
| `R_idx` | `(B, L)` | Residue indices (per-chain) |
| `chain_labels` | `(B, L)` | Numeric chain identifiers |
| `residue_mask` | `(B, L)` | Residue validity mask |
| `designed_residue_mask` | `(B, L)` | Which residues to predict (1=design) |
| `fixed_residue_mask` | `(B, L)` | Complementary partner used as conditioning context |

### LigandMPNN additional features

| Feature | Shape | Description |
|---------|-------|-------------|
| `Y` | `(B, N, 3)` | Non-protein atom coordinates |
| `Y_m` | `(B, N)` | Non-protein atom mask |
| `Y_t` | `(B, N)` | Atom types (element indices) |

For PPI fine-tuning with LigandMPNN, `Y/Y_m/Y_t` include:
- Fixed-partner side-chain atoms (via side-chain atomization from `xyz_37`)
- Glycans, metals, cofactors, and other non-protein atoms present in the structure

Designed-partner side-chain atoms are not exposed through `Y`, so the model
conditions on the partner context without leaking the target residue identities
it is being trained to predict.

### Collation

Variable-length structures are batched in two stages:

1. A **token-budget batch sampler** groups dataset indices until the total
   residue count reaches the token budget (default: 10,000 for ProteinMPNN,
   6,000 for LigandMPNN).
2. A padding collator receives that pre-grouped batch and pads tensors to the
   longest structure in the batch.

This keeps the batching design implementable with standard `DataLoader`
semantics while still handling the wide length distribution efficiently.

## Training Pipeline

### Overview

```
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│ Load pretrained  │────▶│ Fine-tune on PPI data │────▶│ Export checkpoint │
│ weights          │     │                      │     │ (Foundry-compat)  │
└─────────────────┘     └──────────────────────┘     └──────────────────┘
```

Full-model fine-tuning from pretrained ProteinMPNN or LigandMPNN checkpoints.
No frozen layers, no adapter heads, no LoRA — the models are small enough
(1.66M / 2.62M params) to fine-tune entirely.

The default training unit is a partner-design view rather than an all-residue
full-complex redesign. Each complex contributes one view per design direction,
with one partner masked for prediction and the other exposed as fixed context.

### Loss

**Primary:** Label-smoothed negative log-likelihood on amino acid prediction:

```
L = -(1/|M|) * sum_{i in M} [ (1-eps) * log p(s_i | X, s_{<i}) + eps/K * sum_k log p(k | X, s_{<i}) ]
```

where M is the set of designed residue positions, eps=0.1 is the label smoothing
factor, and K=21 is the vocabulary size. In implementation, we accumulate the
masked numerator and masked denominator separately and divide after reduction,
so the loss scale is the mean over designed positions even under variable
token-budget batches and DDP.

**Future extensions** (not in initial implementation):
- Interface residue upweighting (higher loss weight for positions within 8A of
  partner chain)
- Auxiliary contact prediction loss
- ddG-aware loss (MSE between predicted and experimental ddG, as in StaB-ddG)

### Optimizer and scheduler

- **Optimizer:** Adam (beta1=0.9, beta2=0.98, epsilon=1e-9)
- **Scheduler:** Noam (warmup + inverse square root decay)
  - `lr(step) = factor * d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))`
  - Default: `factor=2`, `warmup_steps=4000`
- **Gradient clipping:** None for ProteinMPNN, `max_norm=1.0` for LigandMPNN

### Data augmentation

- **Coordinate noise:** Gaussian noise added to backbone coordinates during
  training. Default: 0.20 A (ProteinMPNN), 0.10 A (LigandMPNN).
- **Random decoding order:** Inherent to the autoregressive formulation — each
  training step uses a different random permutation.

### Training configuration

```python
@dataclass
class TrainingConfig:
    # Model
    model_type: Literal["protein_mpnn", "ligand_mpnn"]
    pretrained_weights: Path

    # Data
    data_sources: list[DataSourceConfig]
    token_budget: int = 10_000
    max_residues: int = 6_000
    min_interface_contacts: int = 4

    # Optimization
    learning_rate_factor: float = 2.0
    warmup_steps: int = 4_000
    max_steps: int = 300_000
    grad_clip_max_norm: float | None = None

    # Augmentation
    structure_noise: float = 0.2
    label_smoothing: float = 0.1

    # Infrastructure
    mixed_precision: bool = True          # bf16 on Ampere+/Blackwell
    gradient_checkpointing: bool = True
    num_workers: int = 8
    seed: int = 42

    # Logging
    log_every_n_steps: int = 100
    eval_every_n_steps: int = 5_000
    save_every_n_steps: int = 10_000
```

### Checkpointing

Each training checkpoint saves a teddyMPNN-native bundle:

```python
{
    "format_version": "teddympnn.v1",
    "model_family": "protein_mpnn" | "ligand_mpnn",
    "state_dict": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "step": global_step,
    "config": config.model_dump(),
    "metrics": {"nll": ..., "seq_recovery": ..., ...},
    "compatibility": {
        "token_order": "foundry_current",
        "rbf_pair_order": "foundry_current",
        "atom_type_vocabulary": "foundry_current",
        "positional_encoding_classes": 66,
    },
}
```

Foundry-compatible checkpoints are exported explicitly from this bundle rather
than used as the primary on-disk training format.

## Evaluation

### 1. Interface sequence recovery

For each structure in the test set:
- Fix all positions outside the interface (within 8A CB-CB of the partner chain)
- Predict sequences at interface positions using teacher forcing
- Compute per-residue accuracy (argmax vs. ground truth)
- Report macro-averaged (per-structure) and micro-averaged (per-residue) recovery

We report both overall recovery and interface-only recovery, stratified by:
- Dataset source (teddymer, NVIDIA, PDB)
- Interface size (small/medium/large)
- CATH classification (for teddymer)

### 2. Binding affinity prediction (ddG)

Following the thermodynamic decomposition from StaB-ddG and BA-DDG:

```
ddG(s_wt → s_mut) ≈ [log p(s_mut | X_AB) - log p(s_mut | X_A) - log p(s_mut | X_B)]
                    - [log p(s_wt  | X_AB) - log p(s_wt  | X_A) - log p(s_wt  | X_B)]
```

This requires **6 forward passes** per mutation: scoring wild-type and mutant
sequences each on the complex structure (X_AB), chain A alone (X_A), and chain B
alone (X_B).

**Variance reduction:**
- **Antithetic variates:** Use identical random decoding orders and backbone noise
  for wild-type and mutant scoring within each sample.
- **Monte Carlo averaging:** Average over M=20 independent samples of decoding
  order and noise.

**Benchmark:** SKEMPI v2.0 (7,085 binding ddG measurements across 345 complexes).
Metrics: per-structure Spearman correlation, overall Spearman/Pearson, RMSE.

**Optional supervised fine-tuning for ddG** (future work):
- Stage 1: Fine-tune on Megascale folding stability data (~776K measurements)
- Stage 2: Fine-tune on SKEMPI v2.0 binding ddG data
- KL divergence penalty against the pretrained model to prevent catastrophic
  forgetting (as in BA-DDG)

### Evaluation API

```python
from teddympnn.evaluation import score_complex, predict_ddg

# Score a complex (log-likelihood per residue)
scores = score_complex(
    model,
    structure_path="complex.pdb",
    designed_chains=["B"],
    num_samples=20,
)

# Predict ddG for a mutation
ddg = predict_ddg(
    model,
    structure_path="complex.pdb",
    mutations={"B": {"A45G": None, "L52W": None}},
    num_samples=20,
)
```

## Package Structure

```
src/teddympnn/
├── __init__.py
├── cli.py                          # Typer CLI (train, evaluate, score, download)
├── config.py                       # Pydantic config models
│
├── models/
│   ├── __init__.py
│   ├── protein_mpnn.py             # ProteinMPNN nn.Module
│   ├── ligand_mpnn.py              # LigandMPNN nn.Module (extends ProteinMPNN)
│   └── layers/
│       ├── __init__.py
│       ├── message_passing.py      # EncLayer, DecLayer
│       ├── graph_embeddings.py     # ProteinFeatures, ProteinFeaturesLigand
│       ├── positional_encoding.py  # PositionalEncodings
│       └── feed_forward.py         # PositionWiseFeedForward
│
├── data/
│   ├── __init__.py
│   ├── dataset.py                  # PPIDataset (unified torch Dataset)
│   ├── sampler.py                  # Token-budget batch sampler
│   ├── collator.py                 # Padding collator for pre-grouped batches
│   ├── features.py                 # Feature computation from PDB/mmCIF
│   ├── teddymer.py                 # Teddymer download + preprocessing
│   ├── nvidia_complexes.py         # NVIDIA complexes download + filtering
│   └── pdb_complexes.py            # PDB experimental complexes
│
├── training/
│   ├── __init__.py
│   ├── trainer.py                  # Training loop (supports DDP)
│   ├── loss.py                     # LabelSmoothedNLLLoss
│   └── scheduler.py                # NoamScheduler
│
├── evaluation/
│   ├── __init__.py
│   ├── sequence_recovery.py        # Interface sequence recovery metrics
│   ├── binding_affinity.py         # ddG prediction (StaB/BA-DDG approach)
│   └── skempi.py                   # SKEMPI v2.0 benchmark utilities
│
└── weights/
    ├── __init__.py
    ├── io.py                       # Native checkpoint bundle I/O
    ├── foundry.py                  # Foundry import/export
    └── legacy.py                   # Legacy ↔ current weight conversion
```

### CLI

```bash
# Download and prepare datasets
teddympnn download teddymer --output /data/teddymer/ --workers 50
teddympnn download nvidia-complexes --output /data/nvidia/ --min-ipsae 0.6
teddympnn download pretrained --model ligand_mpnn --output /weights/
teddympnn checkpoints export-foundry --checkpoint /weights/run.ckpt --output /weights/foundry.pt

# Train
teddympnn train --config configs/ligand_mpnn_ppi.yaml

# Evaluate
teddympnn evaluate recovery --checkpoint /weights/teddympnn_ligand.pt --data /data/test/
teddympnn evaluate ddg --checkpoint /weights/teddympnn_ligand.pt --skempi /data/skempi/

# Score a structure
teddympnn score --checkpoint /weights/teddympnn_ligand.pt --pdb complex.pdb --chains B
```

## Dependencies

```toml
[project]
dependencies = [
    "torch>=2.1",
    "pydantic>=2.0",
    "typer>=0.9",
    "rich>=13.0",
    "biopython>=1.80",
    "pdb-tools>=2.5",
    "numpy>=1.24",
    "pandas>=2.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
train = [
    "wandb>=0.15",
    "lightning>=2.0",        # for DDP / multi-GPU
]
```

## Implementation Order

### Phase 1: Model + weights (foundation)

1. Implement `layers/` — message passing, graph embeddings, positional encoding,
   feed forward. These are the leaf modules and can be unit-tested against
   Foundry reference outputs.
2. Implement `ProteinMPNN` and `LigandMPNN` modules, composing the layers.
3. Implement weight I/O (`weights/io.py`, `weights/foundry.py`,
   `weights/legacy.py`). Verify round-trip: load Foundry checkpoint → our model
   → save native bundle → export Foundry checkpoint → reload in Foundry.
4. **Validation gate:** Run Foundry's inference examples through our model, verify
   identical outputs (within fp32 tolerance).

### Phase 2: Data pipeline

5. Implement teddymer download/preprocessing pipeline.
6. Implement NVIDIA complexes metadata filtering + selective download.
7. Implement unified `PPIDataset`, partner-design view expansion, and
   token-budget batching (`sampler.py` + `collator.py`).
8. Implement feature computation (PDB/mmCIF → feature tensors).

### Phase 3: Training

9. Implement `LabelSmoothedNLLLoss`, `NoamScheduler`.
10. Implement `Trainer` with mixed precision, gradient checkpointing, DDP.
11. Training config + CLI integration.
12. **Validation gate:** End-to-end training run on small subset, verify loss
    decreases and recovery improves.

### Phase 4: Evaluation

13. Implement interface sequence recovery metrics.
14. Implement ddG prediction (6-pass scoring + antithetic variates).
15. Implement SKEMPI v2.0 benchmark.
16. CLI integration for evaluation commands.

### Phase 5: Polish + scale

17. Full training runs on teddymer + NVIDIA + PDB mixture.
18. Hyperparameter tuning.
19. Benchmark against vanilla ProteinMPNN/LigandMPNN on interface recovery +
    SKEMPI ddG.
20. Documentation, examples, and release.
