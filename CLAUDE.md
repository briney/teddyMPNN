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
mypy src/

# CLI
python -m teddympnn --help
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

<!-- Update this section as the project develops. Describe the main components,
     how data flows, and any non-obvious design decisions. -->

TODO: Fill in as the project takes shape.
