# teddyMPNN

A message passing neural network fine-tuned for protein-protein interface design.

teddyMPNN extends [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) and
[LigandMPNN](https://github.com/dauparas/LigandMPNN) with fine-tuning on
large-scale protein-protein interaction datasets for improved interface sequence
recovery, de novo binder design, and binding affinity (ddG) prediction.

## Installation

```bash
pip install teddympnn
```

For development:

```bash
git clone https://github.com/briney/teddympnn.git
cd teddympnn
pip install -e ".[dev,data,train]"
```

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

## Training

### 1. Download data

```bash
# Teddymer synthetic dimers (~510K from AFDB domain pairs)
teddympnn download teddymer --output data/teddymer

# NVIDIA predicted complexes (metadata filtering)
teddympnn download nvidia-complexes --output data/nvidia_complexes

# Pretrained weights
teddympnn download pretrained --model protein_mpnn --noise 020
teddympnn download pretrained --model ligand_mpnn --noise 010
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
# Single run from config
teddympnn train --config configs/run1_proteinmpnn_full.yaml

# Resume from checkpoint
teddympnn train --config configs/run1_proteinmpnn_full.yaml \
    --resume outputs/run1_proteinmpnn_full/checkpoints/step_0050000.pt

# Launch all training runs
python scripts/launch_training.py
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
