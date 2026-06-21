# teddyMPNN

A fine-tuned ProteinMPNN for improved protein-protein interface sequence design.

teddyMPNN fine-tunes [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) on
the [teddymer](https://teddymer.steineggerlab.workers.dev/) dataset of predicted
protein dimers with an interface-weighted cross-entropy loss. The result is a
drop-in ProteinMPNN replacement that picks better residues at interfaces for
tasks like affinity maturation and interface redesign.

## Installation

Install the latest release from PyPI:

```bash
pip install teddympnn
```

The ProteinMPNN base weights ship inside the package, so inference and
fine-tuning work immediately — no separate download step is required.

For data-download extras (`aiohttp`, `zstandard`) add `data`; for training
monitoring (`wandb`) add `train`:

```bash
pip install "teddympnn[data,train]"
```

### From source (development)

```bash
git clone https://github.com/briney/teddympnn.git
cd teddympnn
pip install -e ".[dev]"
```

The editable install is required before running `pytest`, `ty`, or the
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
    --data data/manifests/val_manifest.tsv
```

### Evaluate binding affinity on SKEMPI v2.0

```bash
teddympnn evaluate ddg \
    --checkpoint weights/step_0300000.pt \
    --skempi data/skempi \
    --num-samples 20
```

## Pretrained base weights

The ProteinMPNN base checkpoint (`proteinmpnn_v_48_020.pt`, 48-neighbor,
0.20 Å noise) is bundled with the package and used as the default fine-tuning
starting point. After `pip install` it is available immediately — no separate
download step is required.

The bundled file is redistributed under MIT from
[dauparas/ProteinMPNN](https://github.com/dauparas/ProteinMPNN); see
`src/teddympnn/weights/pretrained/NOTICES.md` for full attribution and
citations.

## Training

### 1. Download teddymer

```bash
teddympnn download teddymer --output data/teddymer
```

### 2. Prepare train/val manifests

```bash
teddympnn download prepare-manifests \
    --output data/manifests \
    --teddymer data/teddymer/filtered_manifest.tsv \
    --val-fraction 0.05
```

### 3. Train

```bash
# Default run (uses configs/train.yaml)
teddympnn train

# Override individual knobs Hydra-style
teddympnn train train.interface_weight=3.0 max_steps=100000

# Resume from checkpoint
teddympnn train --resume outputs/train/checkpoints/step_0050000.pt
```

The `interface_weight` config knob scales the loss at interface residues.
`1.0` (default) reproduces standard ProteinMPNN training; values > 1.0
increase interface emphasis.

## Project Structure

```
src/teddympnn/
    models/          # ProteinMPNN and layers
    data/            # Teddymer pipeline, datasets, manifests
    training/        # Trainer, interface-weighted loss, scheduler
    evaluation/      # Sequence recovery, ΔΔG, SKEMPI
    weights/         # Checkpoint I/O, Foundry base-weight loading
    cli.py           # CLI entry points
    config.py        # Pydantic configuration models
configs/             # Training YAML configs
scripts/             # Utility scripts
tests/               # Test suite
docs/                # Architecture and vision docs
```

## Development

```bash
# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Type check
ty check src/

# Run tests
pytest

# Run tests (skip slow)
pytest -m "not slow"
```

## License

MIT
