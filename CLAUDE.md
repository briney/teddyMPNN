# teddyMPNN

A message passing neural network for protein-protein interfaces.

## Quick Reference

```bash
# Install (editable, with dev dependencies)
pip install -e ".[dev]"

# Run tests
pytest

# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Type check
ty check src/

# CLI
teddympnn --help
```

## Project Structure

```
src/teddympnn/     # Main package code
tests/             # Test suite (mirrors src structure)
```

## Code Conventions

- Python 3.11+ — use modern syntax (type unions with `|`, `match` statements, etc.)
- All public functions and classes need docstrings (Google style)
- Type hints on all function signatures
- Tests go in `tests/` mirroring the src structure: `src/foo/bar.py` → `tests/test_bar.py`
- Ruff handles formatting and linting — don't override its defaults beyond pyproject.toml config

## Before Committing

1. `ruff check --fix src/ tests/` — auto-fix lint issues
2. `ruff format src/ tests/` — format code
3. `pytest` — all tests pass
4. Write a meaningful commit message: `<component>: <what changed and why>`

## Architecture

teddyMPNN fine-tunes the bundled `proteinmpnn_v_48_020` base weights on the
teddymer synthetic dimer dataset using an interface-weighted cross-entropy loss.
The model itself is unchanged (3+3 message-passing layers, 128-dim hidden, k=48,
~1.66M params); the fine-tune sharpens interface residue predictions via a
single `interface_weight` config knob (default 1.0 = standard CE). Evaluation
covers interface sequence recovery (held-out teddymer split + PDB experimental
complexes) and ddG correlation on SKEMPI v2.0. See `docs/VISION.md` for scope
and `docs/ARCHITECTURE.md` for implementation details.
