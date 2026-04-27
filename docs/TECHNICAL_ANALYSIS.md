# teddyMPNN Technical Analysis

## Executive Summary

The repository has a broad scaffold for phases 1-4, but it is not ready for
Phase 5 scale-up. The highest-risk gaps are not formatting or local code style;
they are failures in the data/training/evaluation contracts that would either
train on the wrong distribution, skip validation, break LigandMPNN-specific
conditioning, or make Foundry compatibility unproven.

The main blockers are:

- The data acquisition and manifest pipeline does not yet produce the trainable
  source-specific manifests assumed by the configs and architecture.
- The configured 60/20/20 and 80/0/20 data mixtures are not actually enforced.
- LigandMPNN's PPI-specific context path is incomplete: fixed-partner
  side-chain atomization is not wired into training, periodic-table buffers are
  placeholders, and empty ligand context still changes the protein encoding.
- Checkpoint compatibility is internally inconsistent and only lightly tested;
  the hard Foundry equivalence gate is skipped without local reference files.
- CLI training does not construct validation loaders, does not implement DDP,
  and always treats pretrained weights as legacy-format checkpoints.
- Phase 4 evaluation is scaffolded but not complete enough for the advertised
  SKEMPI and multi-model benchmark workflows.

## Verification Performed

- `ruff check src/ tests/`: passed.
- `mypy src/`: failed with 5 errors in `weights/io.py`,
  `evaluation/binding_affinity.py`, and `models/layers/graph_embeddings.py`.
- `pytest`: failed collection because the active environment cannot import the
  package unless the source tree is installed or `PYTHONPATH=src` is set.
- `PYTHONPATH=src pytest`: passed, but 49 tests were skipped, including the
  Foundry equivalence tests, reference-structure parsing tests, and GPU/pretrained
  e2e tests.

The passing test suite should therefore be interpreted as a scaffold check, not
as evidence that the model, data pipeline, or evaluation workflow is ready for
full training.

## Findings

### P0: The Data Acquisition Pipeline Does Not Produce Trainable Manifests

**Issue:** The README and workplan imply that `teddympnn download teddymer`,
`teddympnn download nvidia-complexes`, and the manifest-preparation path produce
usable training data. The implementation stops short of that contract.

Specific gaps:

- `teddympnn download teddymer` downloads metadata and AFDB full-chain PDBs, but
  does not call `chop_and_assemble_dimers`, so it never creates assembled
  two-chain dimer structures.
- `filter_teddymer_clusters` only applies quality filters to the metadata. It
  does not parse TED domain boundaries into the `domain1_chopping` and
  `domain2_chopping` fields required by `chop_and_assemble_dimers`, nor does it
  write the `structure_path`, `chain_A`, `chain_B`, and `source` columns consumed
  by `PPIDataset`.
- `teddympnn download nvidia-complexes` downloads and filters metadata only. It
  does not download required chunk tarballs or extract the passing structures.
- `download_nvidia_chunks(..., workers=4)` is sequential; the `workers` argument
  is unused despite the scale of the chunk downloads.
- `download_pdb_structures` writes a manifest with `pdb_id`, `structure_path`,
  `num_chains`, and `interface_residues`, but no partner chain columns. The
  later normalization defaults missing chain columns to `A`/`B`, which is wrong
  for many PDB entries.

**Impact:** Phase 5 would start from manifests that either cannot be loaded by
`PPIDataset`, point to unassembled structures, or assign incorrect partner
chains. This is a hard blocker for meaningful training.

**Proposed fix:**

1. Make each source pipeline produce a source-specific training manifest with
   exactly the dataset contract: `structure_path`, `chain_A`, `chain_B`,
   `source`, plus source-specific metadata such as cluster/model/PDB IDs.
2. Wire the teddymer CLI through the full metadata -> AFDB download -> domain
   chop -> dimer assembly -> manifest path.
3. Wire the NVIDIA CLI through metadata filtering -> chunk download -> selective
   extraction -> manifest path, and make chunk downloads actually concurrent or
   explicitly document that they are sequential.
4. For PDB data, identify interacting chain pairs or partner groups and emit
   explicit `chain_A`/`chain_B` values instead of defaulting to `A`/`B`.
5. Add an integration test that creates a tiny manifest from each source
   pipeline and verifies `PPIDataset(...)[0]` returns nonempty designed and fixed
   masks.

### P0: Data Source Mixing and Filtering Are Not Implemented Correctly

**Issue:** The architecture depends on configurable dataset mixtures, but the
actual training path does not enforce the configured source weights.

Specific gaps:

- `MixedDataLoader._build_sampler` constructs a `WeightedRandomSampler`, but
  never uses it. The active sampler is a `TokenBudgetBatchSampler` over the
  concatenated lengths, so `DataSourceConfig.weight` has no effect.
- The provided run configs point all sources at the same
  `data/manifests/train_manifest.tsv`. Because `PPIDataset` does not filter by
  the `source` column or `source_type`, the "teddymer", "nvidia", and "pdb"
  datasets are duplicate views of the same manifest.
- `PPIDataset.min_interface_contacts` is stored but never applied, so the
  minimum-interface-contact filter in the configs is inactive.
- `PPIDataset` does not reject views where the design or fixed partner mask is
  empty. Such examples can silently produce zero-loss batches.

**Impact:** A nominal 60/20/20 run is not a 60/20/20 run. It can duplicate data,
ignore source weights, include non-interface complexes, and train on invalid
views. That directly compromises any Phase 5 benchmark interpretation.

**Proposed fix:**

1. Choose one clear data-loading design:
   either source-specific manifests per `DataSourceConfig`, or one unified
   manifest with an explicit source filter in `PPIDataset`.
2. Replace `MixedDataLoader` with a source-aware batch sampler that actually
   samples according to configured weights while still respecting token budgets.
3. Enforce `min_interface_contacts` during dataset indexing or manifest
   preparation, and fail fast on empty designed/fixed masks.
4. Add tests that sample hundreds of examples from a mixed loader and assert the
   observed source proportions are close to configured weights.

### P0: LigandMPNN PPI Context Diverges From the Architecture

**Issue:** The architecture's main reason for using LigandMPNN in PPI training
is fixed-partner side-chain atomization and richer non-protein context. That
path is currently incomplete or incorrect.

Specific gaps:

- `Trainer.from_config` constructs LigandMPNN datasets with
  `include_ligand_atoms=True`, but leaves `atomize_partner_sidechains=False`.
  Fixed-partner side-chain atoms are therefore not exposed during training.
- The planned train/eval side-chain atomization policy is missing. There is no
  implementation of the random training reveal probability, and inference does
  not explicitly reveal fixed-partner side chains.
- `ProteinFeaturesLigand` registers `side_chain_atom_types`,
  `periodic_table_groups`, and `periodic_table_periods` as all-zero buffers.
  They are never populated in the model constructor. For legacy-loaded LigandMPNN
  weights, the loader intentionally skips missing buffers, leaving all
  group/period features as zero.
- Empty or fully masked ligand context still changes the encoded protein state.
  A direct check showed `max_abs_no_context_delta = 2.34` between the backbone
  encoder output and LigandMPNN's encoded output with `Y_m` all false. The
  workplan expected empty ligand context to match ProteinMPNN behavior.
- `extract_sidechain_atoms` includes atom indices `4:37`, including `OXT`,
  while the architecture specifies 32 side-chain atoms excluding `OXT`.

**Impact:** LigandMPNN fine-tuning will not train the intended PPI conditioning
mechanism. The model can also learn or evaluate through a context branch that
changes representations even when no context exists, making ablations and
ProteinMPNN/LigandMPNN comparisons difficult to interpret.

**Proposed fix:**

1. Populate LigandMPNN periodic-table and side-chain buffers at construction
   time, and verify they match Foundry-current values.
2. Add the side-chain atomization policy to the data/model boundary:
   training uses the configured reveal probabilities, evaluation exposes only
   fixed-partner side chains, and designed-partner atoms are never leaked.
3. Pass `atomize_partner_sidechains=True` for LigandMPNN PPI training unless an
   explicit config disables it.
4. Make the no-context path either exactly bypass the context contribution or
   reproduce Foundry's intended no-context semantics, then add a regression test.
5. Exclude `OXT` from side-chain atomization unless a deliberate terminal-atom
   feature is added to the architecture and compatibility mapping.

### P0: Foundry and Legacy Weight Compatibility Are Not Proven

**Issue:** Weight compatibility is the core Phase 1 promise, but the current
state is internally inconsistent and under-tested.

Specific gaps:

- `docs/ARCHITECTURE.md` and `docs/WORKPLAN.md` list parameter counts of
  1,656,981 for ProteinMPNN and 2,621,973 for LigandMPNN. The implemented tests
  lock in 1,660,485 and 2,618,501 parameters respectively.
- The hard Foundry equivalence tests are skipped unless local reference `.pt`
  files exist. In this checkout they were skipped.
- LigandMPNN validation only checks strict state-dict loading when reference
  data exists; there is no LigandMPNN graph/encoder/decoder output equivalence
  test analogous to ProteinMPNN.
- `Trainer.from_config` always calls `load_legacy_weights`, so it cannot
  correctly auto-load a teddyMPNN-native bundle or Foundry-current checkpoint.
- `configs/benchmark.yaml` lists vanilla pretrained baselines, but
  `evaluation.benchmark._load_model` calls `load_checkpoint_bundle`, which
  expects a teddyMPNN-native bundle and will not load raw legacy/Foundry weights.
- `convert_to_legacy` does not restore LigandMPNN's dropped 120th atom type, so
  reverse export to the original dauparas format is incomplete.

**Impact:** A training run may start from the wrong checkpoint interpretation,
benchmark baselines may fail to load, and exported checkpoints may not be
usable in the ecosystems advertised by the docs. More importantly, Phase 5
results would not be anchored to a verified vanilla ProteinMPNN/LigandMPNN
baseline.

**Proposed fix:**

1. Resolve the authoritative parameter counts by comparing against real Foundry
   models and update either the architecture or implementation.
2. Add required local or CI-accessible Foundry reference artifacts, or make the
   validation script part of an explicit pre-Phase-5 gate that must be run.
3. Add LigandMPNN output equivalence coverage for graph features, context
   encoder outputs, decoder outputs, and no-context behavior.
4. Implement a checkpoint loader dispatcher that detects native teddyMPNN,
   Foundry-current, and legacy checkpoint formats.
5. Update benchmark loading to support pretrained baseline checkpoints directly.
6. Complete or remove the advertised legacy export path for LigandMPNN.

### P0: The Training Orchestration Is Not Phase-5 Ready

**Issue:** The trainer has a plausible single-process loop, but it does not
implement several architecture/workplan requirements needed for full training.

Specific gaps:

- `Trainer.from_config` constructs only a training loader. It never constructs
  or accepts a validation manifest, so CLI training performs no validation even
  though configs include `eval_every_n_steps`.
- DDP is documented in `ARCHITECTURE.md`, `WORKPLAN.md`, and trainer docstrings,
  but the code never initializes `torch.distributed`, wraps the model in
  `DistributedDataParallel`, shards data, or handles ranks.
- LigandMPNN configs leave `grad_clip_max_norm` unset even though the
  architecture calls for `max_norm=1.0`.
- LigandMPNN configs use `token_budget: 10000`, while the architecture
  recommends a smaller LigandMPNN budget of 6,000.
- `Trainer.validate` computes only overall recovery. The interface-recovery
  accumulators are present but never populated.

**Impact:** Phase 5's planned four large runs will not validate during training,
will not scale across multiple GPUs through the advertised path, and will not
track the core interface metric needed to detect regressions.

**Proposed fix:**

1. Extend `TrainingConfig` with explicit training and validation manifest/source
   fields, or make `prepare_manifests` outputs first-class in the config.
2. Build train and validation loaders in `Trainer.from_config`.
3. Implement DDP or remove DDP claims until it exists. If implemented, include
   rank-aware logging, checkpointing, sampler epoch control, and global metric
   reduction.
4. Set LigandMPNN-specific defaults in config validation: `num_neighbors=32`,
   `token_budget=6000`, `structure_noise=0.10`, and `grad_clip_max_norm=1.0`.
5. Reuse `compute_recovery` or equivalent interface-mask logic in validation.

### P1: Token-Budget Batching Does Not Bound Actual Compute or Memory

**Issue:** The architecture uses token-budget batching to keep GPU memory
stable, but the current sampler groups by the sum of unpadded residue counts.
The model then pads to `L_max` and computes dense pairwise distances with
`(B, L_max, L_max, 3)` intermediates before top-k selection.

**Impact:** A batch containing one long complex and several short complexes can
fit the nominal token budget while still performing much more work than
expected because compute scales with `B * L_max^2`, not `sum(L)`. At Phase 5
lengths, this can cause avoidable OOMs or severe throughput collapse.

**Proposed fix:**

1. Add length bucketing or sorting before token-budget packing.
2. Budget against estimated padded compute, such as `B * L_max` for decoder
   memory and `B * L_max^2` for graph construction risk.
3. Consider a separate `max_pairwise_residues` guard for the dense k-NN step.
4. Add sampler tests that construct adversarial long/short mixtures and assert
   bounded padded size.

### P1: Phase 4 Evaluation Is Not Complete Enough for the Advertised Benchmarks

**Issue:** The evaluation modules are useful scaffolds, but important real-data
cases are unsupported or fragile.

Specific gaps:

- LigandMPNN `score`, `predict_ddg`, and SKEMPI paths do not provide `Y`,
  `Y_m`, or `Y_t`, so LigandMPNN ddG/score workflows will fail.
- SKEMPI parsing accepts insertion-code mutations like `L52aG`, but
  `_apply_mutations` converts the residue number substring with `int(...)` and
  `parse_structure` stores only integer residue numbers. Entries with insertion
  codes will be skipped or fail.
- `run_benchmark` accepts multiple recovery test manifests but stores only the
  last recovery result for each model, losing per-test-set metrics.
- Interface recovery is not stratified by dataset source or CATH class as
  described in the architecture.
- The `score_complex` API shown in `ARCHITECTURE.md` is not implemented or
  exported.

**Impact:** The Phase 5 benchmark would under-report failures, skip real SKEMPI
entries, and be unable to compare LigandMPNN ddG performance despite the README
advertising LigandMPNN as a supported model type.

**Proposed fix:**

1. Create a shared structure-to-batch builder that supports both ProteinMPNN and
   LigandMPNN, including ligand extraction and fixed-partner side-chain context.
2. Preserve PDB insertion codes in parsed residue metadata and mutation matching.
3. Store benchmark recovery results per model and per test set.
4. Add real or fixture-based tests for SKEMPI insertion codes and LigandMPNN
   ddG/score paths.
5. Either implement `score_complex` or remove it from the architecture docs.

### P1: The Test Suite Does Not Gate the Critical Risks

**Issue:** The suite has many useful unit tests, but the checks that matter most
for Phase 5 are optional, skipped, or too synthetic.

Specific gaps:

- Foundry equivalence tests were skipped because reference data was absent.
- Reference-structure parsing and dataset tests were skipped because fixture
  structures were absent.
- GPU/pretrained end-to-end tests were skipped because pretrained weights and
  structures were absent.
- `pytest.mark.slow` is not registered in `pyproject.toml`, producing warnings.
- Tests validate that a `WeightedRandomSampler` can be constructed only
  indirectly; they do not verify data-source weighting is honored.
- Tests validate SKEMPI insertion-string parsing but not mutation application
  with insertion codes.
- Tests lock parameter counts that conflict with the architecture spec, so they
  protect the current implementation but not necessarily the intended model.

**Impact:** `PYTHONPATH=src pytest` passing is not a strong signal that phases
1-4 are complete. Major model-compatibility, real-data, and scale-readiness bugs
can survive the current test suite.

**Proposed fix:**

1. Make pre-Phase-5 validation a named test target with required artifacts.
2. Register `slow` and any other custom pytest markers in `pyproject.toml`.
3. Add non-network, committed tiny fixtures for PDB/mmCIF parsing, source
   manifests, source mixing, LigandMPNN no-context behavior, and SKEMPI insertion
   codes.
4. Separate smoke tests from release-gate tests so skipped heavy tests are
   visible and intentional.

### P2: README and Configs Overstate Current User Workflows

**Issue:** The README mostly describes the desired end state, not the current
implementation.

Specific mismatches:

- README data-download commands imply full data preparation, but teddymer and
  NVIDIA commands stop before trainable structures/manifests exist.
- README says pretrained weights can be downloaded with
  `teddympnn download pretrained`, but configs expect `weights/v_48_020.pt` and
  `weights/v_32_010_25.pt`; the downloader writes
  `proteinmpnn_v_48_020.pt` and `ligandmpnn_v_32_010_25.pt`.
- README benchmark config includes vanilla pretrained baselines, but the
  benchmark loader expects teddyMPNN-native checkpoint bundles.
- README describes `pip install teddympnn`; that may be an intended release path,
  but local development currently requires editable installation or
  `PYTHONPATH=src` for tests to import.
- `ARCHITECTURE.md` lists DDP and a Python evaluation API that are not present.

**Impact:** A user following the README can reach failed commands or, worse,
commands that complete without producing the intended data/training behavior.

**Proposed fix:**

1. Split README content into "implemented now" and "planned/Phase 5" sections,
   or update commands only after the underlying workflows are complete.
2. Align pretrained filenames and config paths.
3. Document the required `pip install -e ".[dev,data,train]"` step before
   running tests or CLI commands from a checkout.
4. Add quick smoke commands that are known to pass from a fresh editable install.

### P2: Type-Checking and Minor API Contracts Still Need Cleanup

**Issue:** `mypy src/` currently fails, despite the development workflow
requiring it. Several small config/API fields are also unused.

Specific gaps:

- `mypy src/` reports 5 errors.
- `ModelConfig.num_context_atoms` is accepted but not passed to
  `ProteinFeaturesLigand`; the model uses the module-level
  `NUM_CONTEXT_ATOMS = 25`.
- `score()` docs say the default scores `designed_residue_mask`, but the
  implementation scores all residues unless `score_mask` is provided.

**Impact:** These are not as severe as the data/model/training blockers, but
they make the codebase harder to trust before long-running experiments.

**Proposed fix:**

1. Fix the current mypy errors and keep `mypy src/` in the completion gate.
2. Either wire `num_context_atoms` into LigandMPNN/ProteinFeaturesLigand or
   remove it from config until it is supported.
3. Align `score()` behavior and docstring.

## Phase Completion Status

- **Phase 1: Model and Weight I/O** is partially complete. Core modules exist,
  but Foundry equivalence is not locally validated, LigandMPNN equivalence is
  incomplete, parameter-count documentation conflicts with tests, and checkpoint
  format dispatch is missing.
- **Phase 2: Data Pipeline** is substantially incomplete for real training.
  Parsing and basic loaders exist, but source acquisition does not produce
  final trainable manifests, source mixing is broken, partner-chain assignment is
  fragile, and key filters are inactive.
- **Phase 3: Training** is partially complete for single-process smoke
  training. It lacks validation loader construction, DDP, LigandMPNN-specific
  training defaults, interface validation metrics, and robust checkpoint-format
  loading.
- **Phase 4: Evaluation** is partially complete. Recovery and ddG scaffolds
  exist, but LigandMPNN scoring/ddG is broken, SKEMPI insertion-code handling is
  incomplete, multi-test benchmark reporting loses data, and advertised APIs are
  missing.

## Recommended Fix Order Before Phase 5

1. Resolve model/weight compatibility first: authoritative parameter counts,
   populated LigandMPNN buffers, checkpoint-format dispatch, and Foundry
   equivalence for both ProteinMPNN and LigandMPNN.
2. Finish source-specific data preparation and manifest contracts, including
   teddymer assembly, NVIDIA extraction, PDB partner-chain assignment, and
   source-specific train/validation manifests.
3. Replace `MixedDataLoader` with a tested source-weighted, length-aware
   batcher that bounds padded compute.
4. Wire LigandMPNN side-chain atomization into training/evaluation and add
   no-context regression tests.
5. Complete training orchestration: validation loaders, interface metrics,
   LigandMPNN defaults, DDP or explicit single-GPU-only documentation.
6. Harden Phase 4 benchmarks with LigandMPNN batch construction, SKEMPI
   insertion-code support, and per-test-set reporting.
7. Update README/configs only after the corresponding workflows are actually
   executable.
