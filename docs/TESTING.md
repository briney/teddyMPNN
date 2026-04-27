# Testing teddyMPNN

The test suite is split into a default smoke layer (always runs) and several
release-gate layers that require external reference data. This document
explains what each layer covers, how to populate the reference artifacts,
and how to interpret skipped tests.

## Default smoke suite

```bash
pip install -e ".[dev]"
pytest
```

Covers ~300 tests across model layers, data parsing, dataset/sampler/collator
behavior, evaluation utilities, and CLI plumbing. Runs in a few seconds with
no network and no GPU.

The dataset and feature-parsing tests rely on a tiny synthetic two-chain PDB
fixture. It is committed under
`tests/validation/reference_data/structures/` so the suite is self-contained
on a fresh checkout. If you need to regenerate it (e.g. after editing
`scripts/generate_test_fixtures.py`), run:

```bash
python scripts/generate_test_fixtures.py
```

## Release-gate layers

The following test sets are skipped by default because they require external
reference artifacts that are too large to commit. Run them before tagging a
release or claiming Phase-5 readiness.

### Foundry weight/output equivalence

Files: `tests/validation/test_foundry_equivalence.py` (~25 tests, all
guarded by `requires_reference_data`).

Locks ProteinMPNN and LigandMPNN parameter counts and per-stage outputs
(graph features, encoder, decoder, no-context behavior) against reference
`.pt` tensors generated inside Foundry's published model environment.

Populate:

```bash
# Run inside the Foundry container (requires Foundry-current weights):
python scripts/generate_foundry_reference.py
# Output: tests/validation/reference_data/proteinmpnn_*.pt and ligandmpnn_*.pt
```

Once the `.pt` files exist, `pytest tests/validation/` will run them.

### Pretrained end-to-end training

File: `tests/training/test_e2e_training.py` (marked `@pytest.mark.slow`).

Loads vanilla ProteinMPNN/LigandMPNN checkpoints and runs a few training
steps end-to-end on `1BRS` to verify weight loading + forward/backward +
checkpointing.

Populate:

```bash
# Downloads to tests/validation/reference_data/weights/:
python -m teddympnn download pretrained --model proteinmpnn --output tests/validation/reference_data/weights
python -m teddympnn download pretrained --model ligandmpnn --output tests/validation/reference_data/weights

# A real PDB at tests/validation/reference_data/structures/1BRS.pdb is
# also required (the tiny synthetic 1BRS_mini.pdb fixture is not
# sufficient — these tests probe pretrained-weight behavior).
```

Run with:

```bash
pytest tests/training/test_e2e_training.py -m slow
```

## Interpreting `SKIPPED` lines

Skipped tests in `pytest -v` output indicate the gate is **intentionally
inactive** because the prerequisite reference data was not staged. The
skip messages name the script that produces the data. Re-run the suite
after populating the directory and the corresponding tests should
execute.

If a test you expect to run is skipped:

1. Check the skip reason for the missing artifact path.
2. Run the listed `scripts/...` command.
3. Re-run `pytest`.

## Markers

`pytest.mark.slow` is registered in `pyproject.toml` and applied to the
end-to-end training tests. Use `-m slow` to run only those, or
`-m "not slow"` to skip them in the default smoke pass (already the
behavior on a fresh checkout that lacks weights).
