# teddyMPNN

A message passing neural network fine-tuned for protein-protein interface design.

teddyMPNN extends [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) and
[LigandMPNN](https://github.com/dauparas/LigandMPNN) with fine-tuning on
large-scale protein-protein interaction datasets for improved interface sequence
recovery, de novo binder design, and binding affinity (ddG) prediction.

## Installation

teddyMPNN is not yet published to PyPI. Install from source:

```bash
git clone https://github.com/briney/teddympnn.git
cd teddympnn
pip install -e ".[dev,data,train]"
```

The editable install is required before running `pytest`, `mypy`, or the
`teddympnn` CLI from a checkout — the test suite imports the installed
`teddympnn` package, not the `src/` directory.

## Quick Start

### Score a structure

```bash
teddympnn score \
    --checkpoint weights/step_0300000.pt \
    --pdb structure.pdb \
    --chains A \
    --num-samples 10
```

### Evaluate interface sequence recovery

```bash
teddympnn evaluate recovery \
    --checkpoint weights/step_0300000.pt \
    --data data/manifests/val_manifest.tsv \
    --model-type protein_mpnn
```

### Evaluate binding affinity on SKEMPI v2.0

```bash
teddympnn evaluate ddg \
    --checkpoint weights/step_0300000.pt \
    --skempi data/skempi \
    --num-samples 20
```

### Run multi-model benchmarks

```bash
teddympnn evaluate benchmark \
    --config configs/benchmark.yaml \
    --output results/benchmark.json
```

## Pretrained weights

The standard ProteinMPNN and LigandMPNN base checkpoints are bundled with the
package and used as the default fine-tuning starting point. After
`pip install` they are available immediately — no separate download step is
required.

| `model_type` | Bundled file | Variant |
|---|---|---|
| `protein_mpnn` | `proteinmpnn_v_48_020.pt` | 48 hidden dim, 0.20 Å backbone noise |
| `ligand_mpnn` | `ligandmpnn_v_32_010_25.pt` | 32 hidden dim, 0.10 Å noise, 25-atom ligand context |

The training config's `pretrained_weights` field defaults from `model_type`. To
use a different checkpoint, pass it explicitly:

```bash
# Use the bundled default (selected by model_type)
teddympnn train model_type=protein_mpnn

# Override with a custom checkpoint
teddympnn train pretrained_weights=/path/to/custom.pt
```

The bundled files are redistributed under MIT from
[dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN) and
[dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN); see
`src/teddympnn/weights/pretrained/NOTICES.md` for full attribution and
citations. Other noise variants (002/010/020/030 for ProteinMPNN,
005/010/020/030 for LigandMPNN) can be downloaded on demand from
`files.ipd.uw.edu` via the existing `teddympnn.weights.io.download_pretrained`
utility.

## Training

### 1. Download data

```bash
# Teddymer synthetic dimers (~510K from AFDB domain pairs).
# Reconstructs full side-chain dimers from TED-domain PDB files using the
# Teddymer archive metadata and representative/source indices.
teddympnn download teddymer --output data/teddymer

# NVIDIA predicted complexes (metadata filtering)
teddympnn download nvidia-complexes --output data/nvidia_complexes
```

### 2. Prepare train/val manifests

```bash
teddympnn download prepare-manifests \
    --output data/manifests \
    --teddymer data/teddymer/filtered_manifest.tsv \
    --nvidia data/nvidia_complexes/filtered_manifest.tsv \
    --pdb data/pdb/manifest.tsv \
    --val-fraction 0.05
```

### 3. Train

```bash
# Default ProteinMPNN run (uses configs/train.yaml)
teddympnn train

# Switch model — pretrained weights and architecture defaults follow
teddympnn train model_type=ligand_mpnn output_dir=outputs/ligand_full

# Override individual knobs Hydra-style
teddympnn train model.hidden_dim=256 max_steps=100000 \
    data.train.teddymer.ratio=0.8 data.train.pdb.ratio=0.2

# Resume from checkpoint
teddympnn train --resume outputs/train/checkpoints/step_0050000.pt
```

### 4. Benchmark

```bash
teddympnn evaluate benchmark \
    --config configs/benchmark.yaml \
    --output results/benchmark.json
```

## Training Configurations

| Run | Base Model | Noise | Data Mix | Purpose |
|-----|-----------|-------|----------|---------|
| 1 | ProteinMPNN v_48_020 | 0.20 | 60/20/20 teddymer/nvidia/pdb | Full model |
| 2 | LigandMPNN v_32_010_25 | 0.10 | 60/20/20 teddymer/nvidia/pdb | Full model |
| 3 | ProteinMPNN v_48_020 | 0.20 | 80/0/20 teddymer/pdb | NVIDIA ablation |
| 4 | LigandMPNN v_32_010_25 | 0.10 | 80/0/20 teddymer/pdb | NVIDIA ablation |

Config files are in `configs/`. Runs 3-4 serve as ablations to measure the
contribution of NVIDIA predicted complexes.

## Project Structure

```
src/teddympnn/
    models/          # ProteinMPNN, LigandMPNN, layers
    data/            # Datasets, data acquisition, manifest splitting
    training/        # Trainer, loss, scheduler
    evaluation/      # Sequence recovery, ddG, SKEMPI, benchmarking
    weights/         # Checkpoint I/O, Foundry compatibility
    cli.py           # CLI entry points
    config.py        # Pydantic configuration models
configs/             # Training and benchmark YAML configs
scripts/             # Utility scripts
tests/               # Test suite
docs/                # Architecture and workplan docs
```

## Checkpoint Compatibility

teddyMPNN maintains bidirectional weight compatibility with
[Foundry](https://github.com/dauparas/ProteinMPNN) checkpoints. You can:

- Load pretrained IPD weights directly for fine-tuning
- Export fine-tuned teddyMPNN checkpoints back to Foundry format

```bash
teddympnn checkpoints export-foundry \
    --checkpoint outputs/run1/checkpoints/step_0300000.pt \
    --output foundry_compatible.pt \
    --model-type protein_mpnn
```

## Development

```bash
# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/

# Run tests
pytest

# Run tests (skip slow)
pytest -m "not slow"
```

## License

MIT
