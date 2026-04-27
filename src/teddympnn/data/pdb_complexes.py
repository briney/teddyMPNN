"""PDB experimental complexes data acquisition pipeline.

Curates multi-chain experimental protein structures from the RCSB PDB for
supplementary training data, following Foundry's existing quality filters.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import torch
from rich.progress import Progress

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.cif"

DEFAULT_MAX_RESOLUTION = 3.5
DEFAULT_MIN_CHAINS = 2
DEFAULT_MIN_INTERFACE_CONTACTS = 4
DEFAULT_INTERFACE_DISTANCE = 10.0

# RCSB advanced search query template
_SEARCH_QUERY_TEMPLATE = {
    "query": {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined",
                    "operator": "less",
                    "value": DEFAULT_MAX_RESOLUTION,
                },
            },
            {
                "type": "group",
                "logical_operator": "or",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "exptl.method",
                            "operator": "exact_match",
                            "value": "X-RAY DIFFRACTION",
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "exptl.method",
                            "operator": "exact_match",
                            "value": "ELECTRON MICROSCOPY",
                        },
                    },
                ],
            },
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                    "operator": "greater_or_equal",
                    "value": DEFAULT_MIN_CHAINS,
                },
            },
        ],
    },
    "return_type": "entry",
    "request_options": {
        "results_content_type": ["experimental"],
        "return_all_hits": True,
    },
}


def _best_interacting_chain_pair(
    parsed: dict[str, object],
    *,
    interface_distance: float,
) -> tuple[str, str, int] | None:
    """Find the chain pair with the largest number of interface residues."""
    chain_ids = parsed["chain_ids"]
    assert isinstance(chain_ids, list)
    unique_chain_ids = sorted(set(str(c) for c in chain_ids))
    if len(unique_chain_ids) < 2:
        return None

    from teddympnn.data.features import identify_interface_residues

    best: tuple[str, str, int] | None = None
    for i, chain_a in enumerate(unique_chain_ids):
        for chain_b in unique_chain_ids[i + 1 :]:
            pair_mask = torch.tensor([cid in {chain_a, chain_b} for cid in chain_ids])
            if pair_mask.sum().item() == 0:
                continue
            pair_interface = identify_interface_residues(
                parsed["xyz_37"][pair_mask],
                parsed["xyz_37_m"][pair_mask],
                parsed["chain_labels"][pair_mask],
                distance_cutoff=interface_distance,
            )
            n_interface = int(pair_interface.sum().item())
            if best is None or n_interface > best[2]:
                best = (chain_a, chain_b, n_interface)
    return best


# ---------------------------------------------------------------------------
# Step 1: Query PDB
# ---------------------------------------------------------------------------


def query_pdb_complexes(
    output_path: str | Path,
    *,
    max_resolution: float = DEFAULT_MAX_RESOLUTION,
    min_protein_entities: int = DEFAULT_MIN_CHAINS,
) -> list[str]:
    """Query RCSB PDB for multi-chain protein complexes.

    Uses the RCSB search API to find structures matching resolution and
    chain count criteria.

    Args:
        output_path: Path to write the list of PDB IDs.
        max_resolution: Maximum resolution in Angstroms.
        min_protein_entities: Minimum number of protein entities.

    Returns:
        List of PDB IDs matching the query.
    """
    import urllib.request

    output_path = Path(output_path)

    # Build query
    query = json.loads(json.dumps(_SEARCH_QUERY_TEMPLATE))
    query["query"]["nodes"][0]["parameters"]["value"] = max_resolution
    query["query"]["nodes"][2]["parameters"]["value"] = min_protein_entities

    # Execute search
    logger.info(
        "Querying RCSB PDB: resolution<%.1f Å, ≥%d protein entities",
        max_resolution,
        min_protein_entities,
    )
    req = urllib.request.Request(
        RCSB_SEARCH_URL,
        data=json.dumps(query).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())

    pdb_ids = [hit["identifier"] for hit in result.get("result_set", [])]
    logger.info("Found %d structures", len(pdb_ids))

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(pdb_ids) + "\n")

    return pdb_ids


# ---------------------------------------------------------------------------
# Step 2: Download and filter structures
# ---------------------------------------------------------------------------


def download_pdb_structures(
    pdb_list_path: str | Path,
    output_dir: str | Path,
    *,
    workers: int = 10,
    min_interface_contacts: int = DEFAULT_MIN_INTERFACE_CONTACTS,
    interface_distance: float = DEFAULT_INTERFACE_DISTANCE,
) -> pd.DataFrame:
    """Download mmCIF files from RCSB and filter for interface contacts.

    Downloads structures and verifies that at least 2 chains share
    sufficient interface contacts.

    Args:
        pdb_list_path: Path to file containing PDB IDs (one per line).
        output_dir: Directory to save mmCIF files.
        workers: Number of concurrent download workers.
        min_interface_contacts: Minimum interface residue contacts.
        interface_distance: CB–CB distance cutoff for contacts.

    Returns:
        Manifest DataFrame with columns: pdb_id, structure_path, num_chains.
    """
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    pdb_list_path = Path(pdb_list_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_ids = [line.strip() for line in pdb_list_path.read_text().splitlines() if line.strip()]
    logger.info("Downloading %d PDB structures", len(pdb_ids))

    def _download_one(pdb_id: str) -> str | None:
        out_path = output_dir / f"{pdb_id}.cif"
        if out_path.exists():
            return pdb_id
        url = RCSB_DOWNLOAD_URL.format(pdb_id=pdb_id)
        try:
            urllib.request.urlretrieve(url, out_path)
            return pdb_id
        except Exception:
            logger.debug("Failed to download %s", pdb_id)
            return None

    downloaded: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool, Progress() as progress:
        task = progress.add_task("Downloading", total=len(pdb_ids))
        for result in pool.map(_download_one, pdb_ids):
            if result is not None:
                downloaded.append(result)
            progress.advance(task)

    logger.info("Downloaded %d / %d structures", len(downloaded), len(pdb_ids))

    # Filter for interface contacts
    from teddympnn.data.features import parse_structure

    records: list[dict[str, str | int]] = []
    for pdb_id in downloaded:
        cif_path = output_dir / f"{pdb_id}.cif"
        try:
            parsed = parse_structure(cif_path)
            chain_labels = parsed["chain_labels"]
            n_chains = chain_labels.unique().numel()
            if n_chains < 2:
                continue

            best_pair = _best_interacting_chain_pair(
                parsed,
                interface_distance=interface_distance,
            )
            if best_pair is None:
                continue
            chain_a, chain_b, n_contacts = best_pair
            if n_contacts >= min_interface_contacts:
                records.append(
                    {
                        "pdb_id": pdb_id,
                        "structure_path": str(cif_path),
                        "chain_A": chain_a,
                        "chain_B": chain_b,
                        "source": "pdb",
                        "source_id": pdb_id,
                        "split_group": pdb_id,
                        "num_chains": n_chains,
                        "interface_residues": n_contacts,
                    }
                )
        except Exception:
            logger.debug("Failed to parse %s", pdb_id, exc_info=True)

    manifest = pd.DataFrame(records)
    manifest_path = output_dir / "manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    logger.info("Wrote manifest with %d structures to %s", len(manifest), manifest_path)
    return manifest
