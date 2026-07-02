# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import numpy as np
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align
from biotite.application_v2.tantan import TantanApp
from tests.util import is_not_installed


@pytest.fixture
def simple_matrix():
    alph = seq.NucleotideSequence.alphabet_unamb
    return align.SubstitutionMatrix(
        alph,
        alph,
        np.array([[1, -1, -1, -1], [-1, 1, -1, -1], [-1, -1, 1, -1], [-1, -1, -1, 1]]),
    )


@pytest.mark.skipif(is_not_installed("tantan"), reason="tantan is not installed")
@pytest.mark.parametrize("use_custom_matrix", [False, True])
def test_nucleotide(simple_matrix, use_custom_matrix):
    """
    Test masking a nucleotide sequence based on a known example.
    """
    seq_string = "TGCAAGCTATTAGGCTTAGGTCAGTGCttaagcttaggtcagtgcAACATA"
    sequence = seq.NucleotideSequence(seq_string)

    matrix = simple_matrix if use_custom_matrix else None

    masks = TantanApp().run([sequence], matrix=matrix).result()
    test_mask = masks[0]

    ref_mask = [char.islower() for char in seq_string]
    assert test_mask.tolist() == ref_mask


@pytest.mark.skipif(is_not_installed("tantan"), reason="tantan is not installed")
@pytest.mark.parametrize("use_custom_matrix", [False, True])
def test_protein(use_custom_matrix):
    """
    Test masking a protein sequence based on a known example.
    """
    seq_string = "MAPKINASekinasekinase"
    sequence = seq.ProteinSequence(seq_string)

    matrix = (
        align.SubstitutionMatrix.std_protein_matrix() if use_custom_matrix else None
    )

    masks = TantanApp().run([sequence], matrix=matrix).result()
    test_mask = masks[0]

    ref_mask = [char.islower() for char in seq_string]
    assert test_mask.tolist() == ref_mask


@pytest.mark.skipif(is_not_installed("tantan"), reason="tantan is not installed")
def test_multiple_sequences():
    """
    Masking multiple sequences at once yields one mask per sequence.
    """
    sequences = [
        seq.NucleotideSequence("GGCATCGATATATATATATAGTCAA"),
        seq.NucleotideSequence("ACGTACGTACGT"),
    ]
    masks = TantanApp().run(sequences).result()
    assert len(masks) == 2
    assert [len(mask) for mask in masks] == [len(s) for s in sequences]
