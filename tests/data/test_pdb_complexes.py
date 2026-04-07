"""Tests for PDB experimental complexes data acquisition pipeline."""

from __future__ import annotations

import json

from teddympnn.data.pdb_complexes import _SEARCH_QUERY_TEMPLATE


class TestSearchQuery:
    def test_query_structure(self):
        """Verify the search query template has the expected structure."""
        query = json.loads(json.dumps(_SEARCH_QUERY_TEMPLATE))

        assert query["return_type"] == "entry"
        assert query["query"]["type"] == "group"
        assert query["query"]["logical_operator"] == "and"

        nodes = query["query"]["nodes"]
        assert len(nodes) == 3

        # Resolution filter
        assert nodes[0]["parameters"]["attribute"] == "rcsb_entry_info.resolution_combined"
        assert nodes[0]["parameters"]["operator"] == "less"

        # Method filter (OR group)
        assert nodes[1]["type"] == "group"
        assert nodes[1]["logical_operator"] == "or"

        # Protein entity count filter
        assert nodes[2]["parameters"]["operator"] == "greater_or_equal"
