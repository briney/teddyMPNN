# teddyMPNN Work Plan

Detailed implementation plan for teddyMPNN, organized into phases with concrete
tasks, file outputs, and acceptance criteria.

See [ARCHITECTURE.md](ARCHITECTURE.md) for design rationale and high-level
context.

---

## Phase 1: Model and Weight I/O

Reimplement ProteinMPNN and LigandMPNN with exact weight compatibility.

### 1.1 — Leaf layers

Implement the four leaf modules that compose the larger models.

**Files:**
- `src/teddympnn/models/layers/feed_forward.py`
- `src/teddympnn/models/layers/positional_encoding.py`
- `src/teddympnn/models/layers/message_passing.py`
- `src/teddympnn/models/layers/__init__.py`
- `tests/models/layers/test_feed_forward.py`
- `tests/models/layers/test_positional_encoding.py`
- `tests/models/layers/test_message_passing.py`

**PositionWiseFeedForward:**
- Two-layer MLP: `W_in` (H→4H) → GELU → `W_out` (4H→H)
- Attribute names: `W_in`, `W_out`, `act` (matching Foundry)

**PositionalEncodings:**
- Computes relative residue offsets, clips to ±32, one-hots (65 classes: 64
  offsets + 1 inter-chain), projects through `embed_positional_features` linear
  layer to 16-dim
- `num_positional_features = 2 * max_relative_feature + 1 + 1 = 65`
- Inter-chain pairs get index 64 (the last class)

**EncLayer:**
- Node update: gather neighbor node+edge features → cat with source node →
  3-layer MLP (W1→GELU→W2→GELU→W3) → sum ÷ scale(30) → residual + LayerNorm →
  FFN → residual + LayerNorm
- Edge update: same pattern with W11/W12/W13 → residual + LayerNorm (no FFN)
- `num_in = 3 * hidden_dim` (source node + neighbor node + edge = H+H+H)
- Attributes: W1, W2, W3, W11, W12, W13, norm1, norm2, norm3, dropout1,
  dropout2, dropout3, dense, act, scale, num_hidden, num_in

**DecLayer:**
- Node update only (no edge update)
- Expects pre-concatenated edge features (edge + neighbor node already cat'd by
  caller)
- `num_in = 4 * hidden_dim` for ProteinMPNN decoder (source + neighbor + edge +
  sequence = H+H+H+H)
- `num_in = 3 * hidden_dim` for LigandMPNN protein-ligand context layers
- `num_in = 2 * hidden_dim` for LigandMPNN ligand subgraph layers
- Attributes: W1, W2, W3, norm1, norm2, dropout1, dropout2, dense, act, scale

**Helper functions** (in message_passing.py):
- `gather_nodes(node_features, neighbor_idx)` → [B, L, K, H]
- `gather_edges(edge_features, neighbor_idx)` → [B, L, K, H]
- `cat_neighbors_nodes(h_nodes, h_edges, E_idx)` → [B, L, K, H_edge + H_node]
  (edges first, then nodes)

**Tests:**
- Each layer: forward pass produces correct output shapes
- EncLayer: outputs h_V [B,L,H] and h_E [B,L,K,H]; mask zeroes masked positions
- DecLayer: outputs h_V only; supports variable num_in
- Gradient flows through all parameters

**Acceptance:** All tests pass. Parameter count for each layer matches Foundry.

### 1.2 — Graph embeddings

Implement ProteinFeatures and ProteinFeaturesLigand, which convert raw
coordinates into graph features (neighbor indices + edge embeddings).

**Files:**
- `src/teddympnn/models/layers/graph_embeddings.py`
- `tests/models/layers/test_graph_embeddings.py`

**ProteinFeatures:**

Constants:
- Backbone atoms: N, CA, C, O (indices 0–3 in the 37-atom representation)
- Virtual CB computed from N, CA, C:
  ```
  b = CA - N; c = C - CA; a = cross(b, c)
  CB = -0.58273431*a + 0.56802827*b - 0.54067466*c + CA
  ```
- Representative atom: CA (used for k-NN)
- 5 atoms total (4 backbone + 1 virtual) → 25 atom pairs → 25×16 = 400 RBF dims

Forward pass:
1. Add Gaussian noise to coordinates (`structure_noise` std dev)
2. Extract backbone atom coords from 37-atom X tensor: [B, L, 4, 3]
3. Compute virtual CB: [B, L, 1, 3]
4. Extract CA coords for k-NN: [B, L, 1, 3]
5. Compute CA-CA pairwise distances, take top-k neighbors → E_idx [B, L, K]
6. Concatenate backbone + virtual → [B, L, 5, 3]
7. Compute pairwise RBF: for each of 25 atom pairs (5×5), compute distance
   between atom_a at residue_i and atom_b at neighbor_j, encode with 16 RBFs
   (Gaussian kernels, centers linearly spaced 2.0–22.0 Å) → [B, L, K, 400]
8. Compute positional offsets from R_idx, gather at neighbor indices, compute
   same-chain mask from chain_labels → positional encoding [B, L, K, 16]
9. Concatenate [positional (16) | RBF (400)] → edge_embedding Linear(416→128)
   → edge_norm LayerNorm → E [B, L, K, 128]

Return: `{"E_idx": E_idx, "E": E}`

Attributes: `positional_embedding` (PositionalEncodings), `edge_embedding`
(Linear, bias=False), `edge_norm` (LayerNorm)

**ProteinFeaturesLigand** (extends ProteinFeatures):

Additional constants:
- 32 side-chain atom names (CG, CG1, CG2, ..., OXT)
- Atom type encoding: one-hot element (119 types) + periodic table group (19
  classes) + period (8 classes) = 146 dims → projected to 64
- num_context_atoms = 25

Additional forward logic (after calling parent):
1. Gather 25 nearest non-protein atoms to each residue's virtual CB →
   ligand_subgraph_Y [B, L, 25, 3], Y_m [B, L, 25], Y_t [B, L, 25]
2. If atomize_side_chains: also gather side-chain atoms as additional context,
   merge with ligand atoms, re-select top 25 by distance
3. Protein-to-ligand edge features: RBF from 5 backbone atoms to each context
   atom (5×16=80) + atom type embedding (64) + angle features (4) = 148 →
   node_embedding Linear(148→128) → node_norm LayerNorm
4. Ligand subgraph nodes: atom type → ligand_subgraph_node_embedding
   Linear(146→128) → ligand_subgraph_node_norm LayerNorm
5. Ligand subgraph edges: inter-atom RBF → ligand_subgraph_edge_embedding
   Linear(16→128) → ligand_subgraph_edge_norm LayerNorm

Return: parent dict + `{"E_protein_to_ligand", "ligand_subgraph_nodes",
"ligand_subgraph_edges", "ligand_subgraph_Y_m"}`

Registered buffers (not learned, copied from model not checkpoint):
- `side_chain_atom_types`: [32] tensor of atom type indices
- `periodic_table_groups`: [119] group assignments
- `periodic_table_periods`: [119] period assignments

**Tests:**
- ProteinFeatures: synthetic coordinates → correct E_idx shape [B,L,K], E shape
  [B,L,K,128]; neighbors are actually nearest by CA distance; RBF values in
  (0,1]; positional encoding distinguishes same-chain vs cross-chain
- ProteinFeaturesLigand: correctly gathers ligand atoms, computes context
  features, masks invalid atoms

**Acceptance:** Feature dimensions match Foundry. Neighbor indices are correct
for a known test structure.

### 1.3 — Token encoding

Define the amino acid vocabulary and atom ordering that match Foundry's current
format.

**Files:**
- `src/teddympnn/models/tokens.py`
- `tests/models/test_tokens.py`

**Contents:**
- `AMINO_ACIDS_3TO1`: mapping of 3-letter to 1-letter codes
- `TOKEN_ORDER`: tuple of 21 tokens in 3-letter alphabetical order
  (ALA, ARG, ASN, ASP, CYS, GLN, GLU, GLY, HIS, ILE, LEU, LYS, MET, PHE, PRO,
  SER, THR, TRP, TYR, VAL, UNK)
- `LEGACY_TOKEN_ORDER`: 1-letter alphabetical
  (ALA, CYS, ASP, GLU, PHE, GLY, HIS, ILE, LYS, LEU, MET, ASN, PRO, GLN, ARG,
  SER, THR, VAL, TRP, TYR, UNK)
- `ATOM_ORDER`: 37 atoms
  (N, CA, C, O, CB, CG, CG1, CG2, OG, OG1, SG, CD, CD1, CD2, ND1, ND2, OD1,
  OD2, SD, CE, CE1, CE2, CE3, NE, NE1, NE2, OE1, OE2, CH2, NH1, NH2, OH, CZ,
  CZ2, CZ3, NZ, OXT)
- `token_to_idx` / `idx_to_token` dicts
- `legacy_to_current_token_permutation()` → index mapping
- `legacy_to_current_rbf_permutation()` → index mapping for the 25 atom pairs

**Tests:**
- Permutation indices are correct for known token/RBF orderings
- Round-trip: legacy → current → legacy is identity

### 1.4 — ProteinMPNN module

Compose layers into the full ProteinMPNN model.

**Files:**
- `src/teddympnn/models/protein_mpnn.py`
- `tests/models/test_protein_mpnn.py`

**Module structure** (attribute names match Foundry exactly):
- `graph_featurization_module`: ProteinFeatures
- `W_e`: Linear(128→128, bias=True)
- `W_s`: Embedding(21, 128)
- `encoder_layers`: ModuleList of 3 EncLayer(128, 384, dropout=0.1)
- `decoder_layers`: ModuleList of 3 DecLayer(128, 512, dropout=0.1)
- `W_out`: Linear(128→21, bias=True)

Key note on `num_in` dimensions:
- EncLayer: `num_in = 3H = 384` (source_node H + neighbor_node H + edge H)
- DecLayer: `num_in = 4H = 512` (source_node H + neighbor_node H + edge H +
  sequence_embedding H)

**Forward pass implementation:**

```
encode(input_features, graph_features) → {"h_V": [B,L,H], "h_E": [B,L,K,H]}
  1. h_V = zeros(B, L, H)
  2. h_E = W_e(graph_features["E"])
  3. mask_E = gather residue_mask at neighbor indices, AND with self mask
  4. For each encoder_layer: h_V, h_E = layer(h_V, h_E, E_idx, mask_V, mask_E)
     (with gradient checkpointing)

setup_causality_masks(input_features, graph_features)
  1. Generate random decoding order from designed_residue_mask
  2. Build causal_mask [B,L,K,1] and anti_causal_mask from decoding order + E_idx
  3. Return {causal_mask, anti_causal_mask, decoding_order, decode_last_mask}

decode_teacher_forcing(...)
  1. h_S = W_s(S)  # embed ground truth sequence
  2. Build h_EXV_encoder_anti_causal:
     cat(zeros, h_E, E_idx) → cat(h_V_enc, ..., E_idx) → × anti_causal_mask
  3. For each decoder_layer:
     h_ESV_decoder = cat(h_V_dec, cat(h_S, h_E, E_idx), E_idx)
     h_ESV = causal_mask * h_ESV_decoder + h_EXV_encoder_anti_causal
     h_V_dec = layer(h_V_dec, h_ESV, mask_V, mask_E)
  4. logits = W_out(h_V_dec)
  5. log_probs = log_softmax(logits / temperature + bias)

decode_auto_regressive(...)
  1. Same setup, but iterate over positions in decoding_order
  2. At each step: embed previously sampled token, run decoder, sample next token
```

**score() method** (for ddG evaluation):
- Takes structure + sequence, returns per-residue log-probabilities
- Uses teacher forcing with designed_residue_mask = all True
- Returns log_probs gathered at ground-truth token indices

**Tests:**
- Forward pass with synthetic data: correct output shapes
- Parameter count = 1,656,981 (match Foundry)
- Teacher forcing and autoregressive produce valid log-probs (sum to ≤ 0)
- Gradient flows through all parameters
- score() returns per-residue values in expected range

### 1.5 — LigandMPNN module

Extend ProteinMPNN with the ligand context encoder.

**Files:**
- `src/teddympnn/models/ligand_mpnn.py`
- `tests/models/test_ligand_mpnn.py`

**Additional attributes:**
- `graph_featurization_module`: ProteinFeaturesLigand (replaces ProteinFeatures)
- `W_protein_to_ligand_edges_embed`: Linear(128→128)
- `W_protein_encoding_embed`: Linear(128→128)
- `W_ligand_nodes_embed`: Linear(128→128)
- `W_ligand_edges_embed`: Linear(128→128)
- `W_final_context_embed`: Linear(128→128, bias=False)
- `final_context_norm`: LayerNorm(128)
- `protein_ligand_context_encoder_layers`: ModuleList of 2 DecLayer(128, 384)
- `ligand_context_encoder_layers`: ModuleList of 2 DecLayer(128, 256)

Note num_in differences from base DecLayers:
- protein_ligand_context: `num_in = 3H = 384` (source + edge + ligand_node)
- ligand_context: `num_in = 2H = 256` (source + edge only)

**Override of encode():**
1. Call `super().encode()` → get h_V, h_E from protein backbone encoder
2. Embed ligand features: h_E_p2l = W_protein_to_ligand_edges_embed(E_protein_to_ligand)
3. h_V_context = W_protein_encoding_embed(h_V)
4. Embed ligand subgraph: h_nodes = W_ligand_nodes_embed(ligand_subgraph_nodes),
   h_edges = W_ligand_edges_embed(ligand_subgraph_edges)
5. For i in range(2):
   - h_nodes = ligand_context_encoder_layers[i](h_nodes, h_edges, Y_m, Y_m_edges)
   - h_E_context = cat(h_E_p2l, h_nodes)  → [B,L,25,2H]
   - h_V_context = protein_ligand_context_encoder_layers[i](h_V_context, h_E_context, res_mask, Y_m)
6. h_V = h_V + final_context_norm(dropout(W_final_context_embed(h_V_context)))

**Side-chain atomization masks** (override sample_and_construct_masks):
- Training: randomly reveal side chains with probability
  `overall_atomize_side_chain_probability` (0.5) × per-residue probability (0.02)
- Inference: reveal side chains only for fixed (non-designed) residues
- Revealed side-chain atoms are excluded from loss computation

**Tests:**
- Forward pass with synthetic protein + ligand data: correct shapes
- Parameter count = 2,621,973 (match Foundry)
- Context encoder actually modifies h_V (non-zero contribution)
- With empty ligand context (Y_m all False), output matches ProteinMPNN behavior
- Side-chain atomization masks are correct in train vs eval mode

### 1.6 — Weight I/O and legacy conversion

Load pretrained checkpoints from Foundry (and legacy dauparas repos) and save
fine-tuned checkpoints compatible with both.

**Files:**
- `src/teddympnn/weights/__init__.py`
- `src/teddympnn/weights/io.py`
- `src/teddympnn/weights/legacy.py`
- `tests/weights/test_io.py`
- `tests/weights/test_legacy.py`

**io.py:**
- `load_checkpoint(path, model, strict=True)` → loads either current or legacy
  format, auto-detects by checking for `"model"` vs `"model_state_dict"` key
- `save_checkpoint(path, model, optimizer, scheduler, step, config, metrics)` →
  saves in Foundry-current format (`{"model": state_dict, ...}`)
- `download_pretrained(model_type, noise_level, output_dir)` → downloads from
  `https://files.ipd.uw.edu/pub/ligandmpnn/`

Pretrained checkpoint URLs:
```
ProteinMPNN:  proteinmpnn_v_48_{noise}.pt   noise ∈ {002, 010, 020, 030}
LigandMPNN:   ligandmpnn_v_32_{noise}_25.pt  noise ∈ {005, 010, 020, 030}
```

**legacy.py:**
- `load_legacy_weights(checkpoint, model)` → applies all transformations:
  1. Rename keys (features.* → graph_featurization_module.*, etc.)
  2. Reorder token embeddings in W_s.weight, W_out.weight, W_out.bias
  3. Reorder RBF atom pairs in edge_embedding.weight
  4. Drop 120th atom type (LigandMPNN only)
  5. Copy registered buffers from model (not checkpoint)
- `convert_to_legacy(state_dict)` → reverse transformations for export to
  dauparas repos

Full key renaming map (legacy → current):
```
features.embeddings.linear.{w,b} → graph_featurization_module.positional_embedding.embed_positional_features.{w,b}
features.edge_embedding.weight → graph_featurization_module.edge_embedding.weight
features.norm_edges.{w,b} → graph_featurization_module.edge_norm.{w,b}
features.node_project_down.{w,b} → graph_featurization_module.node_embedding.{w,b}
features.norm_nodes.{w,b} → graph_featurization_module.node_norm.{w,b}
features.type_linear.{w,b} → graph_featurization_module.embed_atom_type_features.{w,b}
features.y_nodes.weight → graph_featurization_module.ligand_subgraph_node_embedding.weight
features.y_edges.weight → graph_featurization_module.ligand_subgraph_edge_embedding.weight
features.norm_y_nodes.{w,b} → graph_featurization_module.ligand_subgraph_node_norm.{w,b}
features.norm_y_edges.{w,b} → graph_featurization_module.ligand_subgraph_edge_norm.{w,b}
W_v.{w,b} → W_protein_to_ligand_edges_embed.{w,b}
W_c.{w,b} → W_protein_encoding_embed.{w,b}
W_nodes_y.{w,b} → W_ligand_nodes_embed.{w,b}
W_edges_y.{w,b} → W_ligand_edges_embed.{w,b}
V_C.weight → W_final_context_embed.weight
V_C_norm.{w,b} → final_context_norm.{w,b}
context_encoder_layers.{i}.* → protein_ligand_context_encoder_layers.{i}.*
y_context_encoder_layers.{i}.* → ligand_context_encoder_layers.{i}.*
```

**Tests:**
- Download proteinmpnn_v_48_020.pt and ligandmpnn_v_32_010_25.pt
- Load into our models with strict=True (no missing/unexpected keys)
- Round-trip: load legacy → save current → reload current → identical state_dict
- convert_to_legacy round-trip: save → convert → load in legacy format

### 1.7 — Validation gate: output equivalence with Foundry

Verify our implementation produces identical outputs to Foundry given the same
weights and inputs. Use the Foundry Docker image as the reference.

**Files:**
- `tests/validation/test_foundry_equivalence.py`
- `tests/validation/conftest.py`
- `scripts/generate_foundry_reference.py`

**Approach:**
1. Write a script that runs inside the Foundry Docker container:
   - Loads a pretrained checkpoint (e.g., proteinmpnn_v_48_020.pt)
   - Processes reference PDB structures (6eb6, 7tdx, 3en2, 2xni)
   - Runs teacher-forcing inference with a fixed seed
   - Saves input tensors and output tensors (log_probs, h_V, E_idx) to .pt files
2. Our test loads the same checkpoint into our model, feeds the same input
   tensors, and asserts torch.allclose on outputs (atol=1e-5 for fp32)
3. Repeat for LigandMPNN with a ligand-containing structure

**Test matrix:**
- ProteinMPNN: 4 reference PDBs × teacher_forcing
- LigandMPNN: at least 1 structure with ligand context
- Both: verify log_probs, encoder h_V, neighbor indices E_idx

**Acceptance:** All outputs match within fp32 tolerance. This is the hard gate
before proceeding to Phase 2.

---

## Phase 2: Data Pipeline

Build the infrastructure to download, preprocess, and serve training data.

### 2.1 — Structure parsing

Parse PDB and mmCIF files into the feature tensor format the models expect.

**Files:**
- `src/teddympnn/data/features.py`
- `tests/data/test_features.py`

**parse_structure(path) → dict:**
1. Read PDB or mmCIF using BioPython (Bio.PDB.PDBParser / MMCIFParser)
2. Extract per-residue 37-atom coordinates → X [L, 37, 3]
3. Extract atom occupancy masks → X_m [L, 37] (bool, True if atom resolved)
4. Encode amino acid sequence → S [L] (using TOKEN_ORDER indices)
5. Compute per-chain residue indices → R_idx [L]
6. Assign numeric chain labels → chain_labels [L]
7. Compute residue validity mask → residue_mask [L]

**extract_ligand_atoms(path) → dict:**
1. Extract non-protein atoms (HETATM records, excluding HOH/NA/CL/K/BR)
2. Map element symbols to atom type indices → Y_t [N]
3. Extract coordinates → Y [N, 3]
4. Validity mask → Y_m [N]

**identify_interface_residues(X, chain_labels, distance_cutoff=8.0) → mask:**
- For each chain pair, find residues with CB-CB distance < cutoff to any residue
  on the other chain
- Returns boolean mask [L]

**Tests:**
- Parse a real PDB file (provide test fixture), verify shapes and values
- Chain labels are correctly assigned for multi-chain structures
- Interface residues are correctly identified
- Ligand atoms extracted with correct element types
- Glycine gets virtual CB at correct position (no real CB)

### 2.2 — Teddymer data acquisition

Download and preprocess teddymer synthetic dimers.

**Files:**
- `src/teddympnn/data/teddymer.py`
- `tests/data/test_teddymer.py`

**Pipeline steps (each a CLI-invocable function):**

1. `download_teddymer_metadata(output_dir)`:
   - Download teddymer.tar.gz from teddymer.steineggerlab.workers.dev
   - Extract cluster.tsv and nonsingletonrep_metadata.tsv
   - Download TED domain boundary table from Zenodo

2. `filter_teddymer_clusters(metadata_dir, output_path)`:
   - Parse nonsingletonrep_metadata.tsv
   - Apply quality filters: InterfacePlddt > 70, AvgIntPAE < 10,
     InterfaceLength > 10
   - Cross-reference with TED domain boundary table to get chopping strings
   - Write filtered manifest (TSV: uniprot_id, domain1_chopping, domain2_chopping,
     cluster_id, interface_length, ...)

3. `download_afdb_structures(manifest_path, output_dir, workers=50)`:
   - Extract unique UniProt IDs from manifest
   - Download full-chain PDBs from
     `https://alphafold.ebi.ac.uk/files/AF-{UNIPROT}-F1-model_v4.pdb`
   - Use asyncio + aiohttp for concurrent downloads
   - Skip already-downloaded files (resume support)
   - Progress bar via rich

4. `chop_and_assemble_dimers(manifest_path, afdb_dir, output_dir, workers=50)`:
   - For each entry in manifest:
     - Chop domain A from full-chain PDB using domain1_chopping
     - Chop domain B using domain2_chopping
     - Relabel chains (A and B)
     - Reset residue numbering per chain
     - Write assembled dimer PDB
   - Use multiprocessing for CPU-bound chopping

**Tests:**
- Parse a sample metadata TSV, verify filter logic
- Chop a known AFDB structure at known domain boundaries, verify residue ranges
- Assembled dimer has two chains with correct residue counts

### 2.3 — NVIDIA complexes data acquisition

Filter and download high-confidence predicted complexes.

**Files:**
- `src/teddympnn/data/nvidia_complexes.py`
- `tests/data/test_nvidia_complexes.py`

**Pipeline steps:**

1. `download_nvidia_metadata(output_dir)`:
   - Download model_entity_metadata_mapping.csv (4.3 GB) from EBI FTP
   - This is a one-time download

2. `filter_nvidia_metadata(csv_path, output_path, min_ipsae=0.6, min_plddt=70, max_clashes=10)`:
   - Read CSV with pandas (chunked for memory efficiency)
   - Apply filters: ipSAEmin >= min_ipsae, pLDDTavg >= min_plddt,
     N_clash_backbone <= max_clashes
   - Group passing structures by chunk tarball → write manifest
   - Report: number of passing structures, number of chunks needed, estimated
     download size

3. `download_nvidia_chunks(manifest_path, output_dir, workers=4)`:
   - Download only the required chunk_NNNN.tar files from FTP
   - Each chunk is ~7.5 GB
   - Use ftplib or urllib for downloads

4. `extract_nvidia_structures(manifest_path, chunks_dir, output_dir)`:
   - For each chunk tarball, extract only the passing structures
   - Decompress zstd-compressed files within tarballs
   - Convert mmCIF → PDB for uniform downstream processing (or parse mmCIF
     directly in features.py)

**Tests:**
- Filter logic on a synthetic metadata CSV
- Chunk mapping is correct (manifest entry → correct tarball)

### 2.4 — PDB experimental complexes

Curate multi-chain experimental structures from the PDB.

**Files:**
- `src/teddympnn/data/pdb_complexes.py`
- `tests/data/test_pdb_complexes.py`

**Pipeline steps:**

1. `query_pdb_complexes(output_path)`:
   - Query RCSB PDB search API for structures matching:
     - Resolution < 3.5 Å
     - Method: X-ray diffraction or cryo-EM
     - ≥ 2 protein entities
   - Write list of PDB IDs + metadata

2. `download_pdb_structures(pdb_list, output_dir, workers=10)`:
   - Download mmCIF files from RCSB
   - Filter to structures where at least 2 chains share interface contacts
     (≥ 4 residues within 10 Å CB-CB)
   - Write final manifest

**Tests:**
- Query produces a non-empty list of PDB IDs
- Downloaded structures have ≥ 2 chains with contacts

### 2.5 — Unified dataset

A single PyTorch Dataset that loads from any of the three data sources.

**Files:**
- `src/teddympnn/data/dataset.py`
- `tests/data/test_dataset.py`

**PPIDataset(torch.utils.data.Dataset):**
- Constructor takes a manifest file (TSV with columns: structure_path, source,
  chain_A, chain_B, interface_residues, ...)
- `__getitem__` returns a feature dict:
  ```python
  {
      "X": Tensor,           # (L, 37, 3)
      "X_m": Tensor,         # (L, 37)
      "S": Tensor,           # (L,)
      "R_idx": Tensor,       # (L,)
      "chain_labels": Tensor, # (L,)
      "residue_mask": Tensor, # (L,)
      "designed_residue_mask": Tensor,  # (L,) — all True for training
      "Y": Tensor,           # (N, 3) — empty for ProteinMPNN
      "Y_m": Tensor,         # (N,)
      "Y_t": Tensor,         # (N,)
      "num_residues": int,   # for token budget batching
  }
  ```
- Supports optional preprocessing cache (parsed features saved as .pt files)
- Supports max_residues filter (skip structures longer than threshold)
- Supports min_interface_contacts filter

**MixedDataLoader:**
- Wraps multiple PPIDatasets with configurable sampling weights
- Each epoch draws from datasets proportionally to their weights
- Uses `torch.utils.data.WeightedRandomSampler` or custom interleaving

**Tests:**
- Dataset returns correct tensor types and shapes
- Mixing weights produce approximately correct sampling ratios
- Caching produces identical outputs to uncached loading
- Structures exceeding max_residues are skipped

### 2.6 — Token-budget collator

Batch variable-length structures by total residue count rather than example
count.

**Files:**
- `src/teddympnn/data/collator.py`
- `tests/data/test_collator.py`

**TokenBudgetCollator:**
- Takes a token budget (default 10,000 for ProteinMPNN, 6,000 for LigandMPNN)
- Groups structures until cumulative residue count reaches budget
- Pads all tensors in the batch to the longest sequence
- Padding values: X=0.0, X_m=False, S=UNK_IDX, R_idx=-100,
  chain_labels=-1, residue_mask=False, designed_residue_mask=False,
  Y=0.0, Y_m=0, Y_t=0

**Integration with DataLoader:**
- Used as the `collate_fn` parameter
- Works with `batch_size=1` in the DataLoader (collator handles grouping)
- Or: pre-group indices into token-budget batches via a custom `BatchSampler`

**Tests:**
- Batch total residues ≤ token budget (±1 structure tolerance)
- Padding is correct and masks are False at padded positions
- Output shapes are [B, L_max, ...] where L_max is the longest in the batch

---

## Phase 3: Training

### 3.1 — Loss function

**Files:**
- `src/teddympnn/training/loss.py`
- `tests/training/test_loss.py`

**LabelSmoothedNLLLoss:**
- `label_smoothing_eps`: 0.1
- `normalization_constant`: 6000.0 (fixed, not per-sample)
- Forward:
  1. One-hot encode target S → [B, L, 21]
  2. Apply label smoothing: `(1-eps) * onehot + eps/21`
  3. Multiply by log_probs, sum over vocab dim → per-residue NLL
  4. Mask by mask_for_loss, sum → divide by normalization_constant

**Tests:**
- Perfect predictions (log_prob=0 at correct token) → loss ≈ eps * log(21) / norm
- Uniform predictions → loss ≈ log(21) / norm
- Masked positions don't contribute to loss
- Gradient flows to log_probs

### 3.2 — Learning rate scheduler

**Files:**
- `src/teddympnn/training/scheduler.py`
- `tests/training/test_scheduler.py`

**NoamScheduler:**
- Wraps `torch.optim.lr_scheduler.LambdaLR`
- Lambda: `factor * d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))`
- Default: factor=2, d_model=128, warmup_steps=4000
- Step 0 returns 0.0 (prevents division by zero)

**Tests:**
- LR is 0 at step 0
- LR increases during warmup (steps 1–4000)
- LR peaks at step ~warmup_steps
- LR decays as step^(-0.5) after warmup

### 3.3 — Trainer

**Files:**
- `src/teddympnn/training/trainer.py`
- `tests/training/test_trainer.py`

**Trainer:**
- Supports single-GPU and multi-GPU (DDP via `torch.distributed`)
- Mixed precision via `torch.amp.autocast` (bf16 on Ampere+/Blackwell, fp16
  fallback) + GradScaler
- Gradient checkpointing (already in model via `torch.utils.checkpoint`)
- Gradient clipping: optional max_norm (1.0 for LigandMPNN, None for ProteinMPNN)

Training loop:
```
for step in range(max_steps):
    batch = next(data_iter)
    
    with autocast(dtype=bf16):
        output = model({"input_features": batch})
        loss, metrics = loss_fn({"input_features": batch}, output, {})
    
    scaler.scale(loss).backward()
    if grad_clip: scaler.unscale_(optimizer); clip_grad_norm_(...)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    optimizer.zero_grad()
    
    if step % log_every == 0: log_metrics(...)
    if step % eval_every == 0: run_validation(...)
    if step % save_every == 0: save_checkpoint(...)
```

Validation loop:
- Run model in eval mode on validation set
- Compute NLL and sequence recovery (overall + interface-only)
- Log metrics

Checkpoint resume:
- Load model, optimizer, scheduler, step from checkpoint
- Resume from step+1

**Tests:**
- e2e integration test: create tiny model (1 encoder layer, 1 decoder layer,
  hidden_dim=32), train on synthetic data for 100 steps on the local A6000
- Loss decreases over training
- Checkpoint save + resume produces identical results
- Mixed precision doesn't produce NaN/Inf
- Gradient clipping caps gradient norm correctly

### 3.4 — Configuration

**Files:**
- `src/teddympnn/config.py`
- `tests/test_config.py`

**Pydantic models:**
```python
class DataSourceConfig(BaseModel):
    name: str
    weight: float
    path: Path
    source_type: Literal["teddymer", "nvidia", "pdb"]

class ModelConfig(BaseModel):
    model_type: Literal["protein_mpnn", "ligand_mpnn"]
    hidden_dim: int = 128
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    num_neighbors: int = 48  # 32 for LigandMPNN
    num_context_atoms: int = 25  # LigandMPNN only
    dropout_rate: float = 0.1

class TrainingConfig(BaseModel):
    model: ModelConfig
    pretrained_weights: Path
    data_sources: list[DataSourceConfig]
    token_budget: int = 10_000
    max_residues: int = 6_000
    min_interface_contacts: int = 4
    learning_rate_factor: float = 2.0
    warmup_steps: int = 4_000
    max_steps: int = 300_000
    grad_clip_max_norm: float | None = None
    structure_noise: float = 0.2
    label_smoothing: float = 0.1
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    num_workers: int = 8
    seed: int = 42
    log_every_n_steps: int = 100
    eval_every_n_steps: int = 5_000
    save_every_n_steps: int = 10_000
    output_dir: Path = Path("outputs")
    wandb_project: str | None = None

class EvalConfig(BaseModel):
    checkpoint: Path
    data_path: Path
    metrics: list[Literal["recovery", "interface_recovery", "ddg"]]
    num_samples: int = 20  # for ddG Monte Carlo averaging
    skempi_path: Path | None = None
```

- Load from YAML: `TrainingConfig.model_validate(yaml.safe_load(open(path)))`
- Sensible defaults for both ProteinMPNN and LigandMPNN

**Tests:**
- Valid YAML loads correctly
- Invalid config raises ValidationError with clear message
- Default values are correct for each model type

### 3.5 — CLI integration

**Files:**
- `src/teddympnn/cli.py`
- `tests/test_cli.py`

**Commands** (via Typer):

```
teddympnn download teddymer --output DIR [--workers N]
teddympnn download nvidia-complexes --output DIR [--min-ipsae FLOAT] [--min-plddt FLOAT]
teddympnn download pretrained --model MODEL --output DIR [--noise FLOAT]
teddympnn train --config PATH
teddympnn evaluate recovery --checkpoint PATH --data PATH
teddympnn evaluate ddg --checkpoint PATH --skempi PATH [--num-samples N]
teddympnn score --checkpoint PATH --pdb PATH --chains STR [--num-samples N]
```

Also register as entry point in pyproject.toml:
```toml
[project.scripts]
teddympnn = "teddympnn.cli:app"
```

**Tests:**
- `--help` works for all commands
- `train` with a minimal config runs without error (using tiny model + synthetic
  data)

### 3.6 — Validation gate: end-to-end training

Run a complete training pipeline on a small dataset using the local A6000.

**Test:** `tests/training/test_e2e_training.py`

Steps:
1. Download proteinmpnn_v_48_020.pt pretrained weights
2. Prepare a small test set: 10 PDB complexes (provide list in test fixtures)
3. Run training for 500 steps with reduced hyperparameters:
   - token_budget=2000, max_steps=500, eval_every=100, save_every=250
4. Assert:
   - Training loss decreases monotonically (smoothed)
   - Validation sequence recovery is > random (> 1/20 = 5%)
   - Checkpoint files are created at steps 250 and 500
   - Checkpoint can be loaded and produces valid outputs
   - No CUDA OOM errors
5. Repeat for LigandMPNN with ligandmpnn_v_32_010_25.pt

Mark with `@pytest.mark.slow` (requires GPU, takes ~5 minutes).

**Acceptance:** Loss decreases, recovery improves, checkpoints are valid, no
errors. This gates the start of Phase 4.

---

## Phase 4: Evaluation

### 4.1 — Interface sequence recovery

**Files:**
- `src/teddympnn/evaluation/sequence_recovery.py`
- `tests/evaluation/test_sequence_recovery.py`

**Metrics:**
- `overall_recovery`: fraction of correct amino acid predictions across all
  designed positions
- `interface_recovery`: same but restricted to interface residues (CB-CB < 8 Å
  to partner chain)
- `per_structure_recovery`: macro-averaged (mean of per-structure recoveries)
- Stratified by: dataset source, interface size bins (1–20, 21–50, 51+
  residues), CATH class (for teddymer)

**compute_recovery(model, dataset, interface_cutoff=8.0) → RecoveryResults:**
1. For each structure: run teacher-forcing, get argmax predictions
2. Compare to ground truth at designed positions
3. Aggregate with and without interface mask

**Tests:**
- Perfect model (identity mapping) → 100% recovery
- Random model → ~4.8% recovery (1/21)
- Interface mask correctly selects interface residues only

### 4.2 — Binding affinity prediction (ddG)

**Files:**
- `src/teddympnn/evaluation/binding_affinity.py`
- `tests/evaluation/test_binding_affinity.py`

**predict_ddg(model, structure_path, mutations, num_samples=20) → float:**

Implementation:
1. Parse the complex structure → feature dict
2. Extract chain A and chain B as separate structures (for monomer scoring)
3. Apply mutation(s) to the sequence tensor (swap token indices)
4. For each of M=20 Monte Carlo samples:
   a. Generate shared random state (decoding order + noise)
   b. Score wt on complex → log_prob_wt_AB
   c. Score mut on complex → log_prob_mut_AB (same random state = antithetic)
   d. Score wt on chain A alone → log_prob_wt_A
   e. Score mut on chain A alone → log_prob_mut_A
   f. Score wt on chain B alone → log_prob_wt_B
   g. Score mut on chain B alone → log_prob_mut_B
   h. Compute binding energy proxy:
      ```
      b_wt  = log_prob_wt_AB  - log_prob_wt_A  - log_prob_wt_B
      b_mut = log_prob_mut_AB - log_prob_mut_A - log_prob_mut_B
      ddg_sample = b_mut - b_wt
      ```
5. Return mean(ddg_samples)

Where log_prob is the sum of per-residue log-probabilities at ground-truth (or
mutant) tokens, restricted to the mutation site(s).

**score_structure(model, features, sequence, designed_mask, rng_state) → float:**
- Helper that runs a single scoring pass with a fixed random state
- Sets model to eval mode
- Uses teacher forcing with the given sequence
- Returns sum of log_probs at designed positions

**Tests:**
- Identity mutation (wt → wt) gives ddG ≈ 0
- Single-residue mutation produces a finite, non-NaN value
- Antithetic variates reduce variance (compare with/without shared random state)
- Multi-residue mutations are handled correctly

### 4.3 — SKEMPI v2.0 benchmark

**Files:**
- `src/teddympnn/evaluation/skempi.py`
- `tests/evaluation/test_skempi.py`

**Pipeline:**
1. Download SKEMPI v2.0 database (CSV) from
   https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv
2. Download associated PDB structures
3. Parse mutations and experimental ddG values
4. For each entry: run predict_ddg with the appropriate structure and mutations
5. Compute correlation metrics:
   - Overall Spearman and Pearson correlation
   - Per-structure Spearman (then report median)
   - RMSE, MAE
   - AUROC for classifying stabilizing vs destabilizing mutations

**evaluate_skempi(model, skempi_dir, num_samples=20) → SKEMPIResults:**
- Returns a dataclass with all metrics + per-structure breakdowns
- Supports filtering by mutation type (single vs multi), interface vs non-interface

**Tests:**
- SKEMPI CSV parsing produces correct mutation/ddG pairs
- At least one structure can be scored end-to-end
- Metrics computation is correct on synthetic predictions

---

## Phase 5: Scale and Benchmark

### 5.1 — Full data preparation

Run the complete data acquisition pipelines from Phase 2:
- Teddymer: ~510K dimers (~125 GB AFDB downloads + ~50 GB processed)
- NVIDIA complexes: filter metadata → download chunks → extract structures
- PDB complexes: query + download

Write unified training/validation manifests with an 95/5 split (by cluster for
teddymer, by complex for NVIDIA, by structure for PDB).

### 5.2 — Training runs

Run four main training configurations:

| Run | Base model | Noise | Data mix |
|-----|-----------|-------|----------|
| 1 | ProteinMPNN v_48_020 | 0.20 | 60/20/20 teddymer/nvidia/pdb |
| 2 | LigandMPNN v_32_010_25 | 0.10 | 60/20/20 teddymer/nvidia/pdb |
| 3 | ProteinMPNN v_48_020 | 0.20 | 80/0/20 teddymer/pdb (no nvidia) |
| 4 | LigandMPNN v_32_010_25 | 0.10 | 80/0/20 teddymer/pdb (no nvidia) |

Runs 3–4 serve as ablation to measure the contribution of the NVIDIA predicted
complexes.

Training parameters: max_steps=300K, token_budget per model defaults, eval every
5K steps. Log to wandb.

### 5.3 — Benchmarking

Compare fine-tuned teddyMPNN models against baselines:

**Interface sequence recovery:**
- Baseline: vanilla ProteinMPNN, vanilla LigandMPNN (pretrained, no fine-tuning)
- Test sets: held-out teddymer clusters, held-out NVIDIA complexes, held-out PDB
  complexes
- Report overall recovery, interface-only recovery, per-structure recovery

**Binding affinity (SKEMPI v2.0):**
- Baselines: vanilla ProteinMPNN, vanilla LigandMPNN, FoldX (if available)
- Run full SKEMPI evaluation with num_samples=20
- Report per-structure Spearman, overall Spearman/Pearson, RMSE

### 5.4 — Documentation and release

- Update README.md with installation, quickstart, and example usage
- Add example configs in `configs/` directory
- Add example scripts in `scripts/` for common workflows
- Ensure all public APIs have docstrings
- Tag release version

---

## Dependency Summary

```toml
[project]
dependencies = [
    "torch>=2.2",
    "pydantic>=2.0",
    "typer>=0.9",
    "rich>=13.0",
    "biopython>=1.80",
    "pdb-tools>=2.5",
    "numpy>=1.24",
    "pandas>=2.0",
    "pyyaml>=6.0",
    "aiohttp>=3.9",         # async downloads
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.4",
    "mypy>=1.0",
    "pre-commit>=3.0",
]
train = [
    "wandb>=0.15",
]
```

---

## Task Dependencies

```
Phase 1:
  1.1 (layers) ─────────────────┐
  1.3 (tokens) ─────────────────┤
                                ├─▶ 1.4 (ProteinMPNN)
  1.2 (graph embeddings) ──────┤   │
                                │   ├─▶ 1.5 (LigandMPNN)
                                │   │   │
                                │   │   ├─▶ 1.6 (weight I/O)
                                │   │   │   │
                                │   │   │   ├─▶ 1.7 (validation gate)
Phase 2:                        │   │   │   │
  2.1 (parsing) ────────────────┤   │   │   │
                                ├─▶ 2.5 (dataset)
  2.2 (teddymer) ──────────────┤   │
  2.3 (nvidia) ────────────────┤   ├─▶ 2.6 (collator)
  2.4 (pdb) ───────────────────┘   │
                                    │
Phase 3:                            │
  3.1 (loss) ──────────────────┐    │
  3.2 (scheduler) ─────────────┤    │
                               ├─▶ 3.3 (trainer) ◀── 2.6
  3.4 (config) ───────────────┤    │
                               │    ├─▶ 3.5 (CLI)
                               │    │
                               │    ├─▶ 3.6 (e2e gate) ◀── 1.7
Phase 4:                       │    │
  4.1 (recovery) ◀── 1.4 ─────┤    │
  4.2 (ddg) ◀── 1.4 ──────────┤    │
  4.3 (skempi) ◀── 4.2 ───────┘    │
                                    │
Phase 5:                            │
  5.1 (data prep) ◀── 2.2-2.4 ─────┘
  5.2 (training) ◀── 3.6, 5.1
  5.3 (benchmark) ◀── 4.1-4.3, 5.2
  5.4 (docs) ◀── 5.3
```

Within each phase, tasks can be parallelized where dependencies allow. In
particular:
- 1.1, 1.2, 1.3 are independent
- 2.2, 2.3, 2.4 are independent
- 3.1, 3.2, 3.4 are independent
- 4.1, 4.2 are independent (both depend on 1.4)
