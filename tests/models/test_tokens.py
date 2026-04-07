"""Tests for amino acid vocabulary and permutation utilities."""

from __future__ import annotations

from teddympnn.models.tokens import (
    AMINO_ACIDS_1TO3,
    AMINO_ACIDS_3TO1,
    ATOM_ORDER,
    BACKBONE_ATOMS,
    LEGACY_TOKEN_ORDER,
    NUM_ATOMS_37,
    NUM_RBF_PAIRS,
    TOKEN_ORDER,
    VOCAB_SIZE,
    current_to_legacy_rbf_permutation,
    current_to_legacy_token_permutation,
    expand_pair_permutation,
    legacy_to_current_rbf_permutation,
    legacy_to_current_token_permutation,
)


class TestTokenOrder:
    def test_vocab_size(self) -> None:
        assert VOCAB_SIZE == 21
        assert len(TOKEN_ORDER) == 21
        assert len(LEGACY_TOKEN_ORDER) == 21

    def test_current_order_is_3letter_alphabetical(self) -> None:
        sorted_tokens = sorted(TOKEN_ORDER[:-1])  # Exclude UNK
        assert list(TOKEN_ORDER[:-1]) == sorted_tokens
        assert TOKEN_ORDER[-1] == "UNK"

    def test_legacy_order_is_1letter_alphabetical(self) -> None:
        one_letters = [AMINO_ACIDS_3TO1[tok] for tok in LEGACY_TOKEN_ORDER[:-1]]
        assert one_letters == sorted(one_letters)
        assert LEGACY_TOKEN_ORDER[-1] == "UNK"

    def test_same_residues_in_both_orders(self) -> None:
        assert set(TOKEN_ORDER) == set(LEGACY_TOKEN_ORDER)

    def test_3to1_and_1to3_are_inverse(self) -> None:
        for three, one in AMINO_ACIDS_3TO1.items():
            assert AMINO_ACIDS_1TO3[one] == three

    def test_num_atoms(self) -> None:
        assert NUM_ATOMS_37 == 37
        assert len(ATOM_ORDER) == 37

    def test_backbone_atoms(self) -> None:
        assert BACKBONE_ATOMS == ("N", "CA", "C", "O", "CB")
        assert NUM_RBF_PAIRS == 25


class TestTokenPermutations:
    def test_legacy_to_current_maps_correctly(self) -> None:
        perm = legacy_to_current_token_permutation()
        assert len(perm) == VOCAB_SIZE
        # Verify: current[i] should equal legacy[perm[i]]
        for i, tok in enumerate(TOKEN_ORDER):
            assert LEGACY_TOKEN_ORDER[perm[i]] == tok

    def test_current_to_legacy_maps_correctly(self) -> None:
        perm = current_to_legacy_token_permutation()
        assert len(perm) == VOCAB_SIZE
        for i, tok in enumerate(LEGACY_TOKEN_ORDER):
            assert TOKEN_ORDER[perm[i]] == tok

    def test_token_permutation_roundtrip(self) -> None:
        """legacy → current → legacy is identity."""
        l2c = legacy_to_current_token_permutation()
        c2l = current_to_legacy_token_permutation()
        # Compose: for each legacy index i, l2c gives current index
        # then c2l[l2c[i]] should give back i... but these permutations
        # are gather-style, so: current[j] = legacy[l2c[j]]
        # legacy[k] = current[c2l[k]]
        # legacy[k] = legacy[l2c[c2l[k]]] → l2c[c2l[k]] = k
        for k in range(VOCAB_SIZE):
            assert l2c[c2l[k]] == k


class TestRBFPermutations:
    def test_legacy_to_current_has_correct_length(self) -> None:
        perm = legacy_to_current_rbf_permutation()
        assert len(perm) == NUM_RBF_PAIRS

    def test_current_to_legacy_has_correct_length(self) -> None:
        perm = current_to_legacy_rbf_permutation()
        assert len(perm) == NUM_RBF_PAIRS

    def test_rbf_permutation_roundtrip(self) -> None:
        l2c = legacy_to_current_rbf_permutation()
        c2l = current_to_legacy_rbf_permutation()
        for k in range(NUM_RBF_PAIRS):
            assert l2c[c2l[k]] == k

    def test_expand_pair_permutation(self) -> None:
        perm = [1, 0]  # Swap first two pairs
        expanded = expand_pair_permutation(perm, num_rbf=3)
        assert expanded == [3, 4, 5, 0, 1, 2]

    def test_legacy_pair_order_matches_foundry(self) -> None:
        """Verify legacy permutation maps known pairs correctly.

        The Foundry legacy ordering starts with CA-CA at index 0 and
        N-N at index 1. In the current outer-product ordering over
        (N, CA, C, O, CB), N-N is at index 0 and CA-CA is at index 6.
        """
        perm = legacy_to_current_rbf_permutation()
        # current[0] = (N,N) should come from legacy[1] (N-N)
        assert perm[0] == 1
        # current[6] = (CA,CA) should come from legacy[0] (Ca-Ca)
        assert perm[6] == 0
        # current[12] = (C,C) should come from legacy[2]
        assert perm[12] == 2
