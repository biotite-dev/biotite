# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import numpy as np
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align
from biotite.application.tantan import TantanApp
from ..util import is_not_installed

@pytest.fixture
def simple_matrix():
    alph = seq.NucleotideSequence.alphabet_unamb
    return align.SubstitutionMatrix(
        alph, alph, np.array(
            [[ 1, -1, -1, -1],
             [-1,  1, -1, -1],
             [-1, -1,  1, -1],
             [-1, -1, -1,  1]]
        )
    )


@pytest.mark.skipif(
    is_not_installed("tantan"), reason="tantan is not installed"
)
@pytest.mark.parametrize("use_custom_matrix", [False, True])
def test_nucleotide(simple_matrix, use_custom_matrix):
    """
    Test masking a nucleotide sequence based on a known example.
    """
    seq_string = "TGCAAGCTATTAGGCTTAGGTCAGTGCttaagcttaggtcagtgcAACATA"
    sequence = seq.NucleotideSequence(seq_string)

    if use_custom_matrix:
        matrix = simple_matrix
    else:
        matrix = None

    test_mask = TantanApp.mask_repeats(sequence, matrix)

    ref_mask = [True if char.islower() else False for char in seq_string]

    assert len(test_mask) == len(ref_mask)
    assert np.all(test_mask.tolist() == ref_mask)


@pytest.mark.skipif(
    is_not_installed("tantan"), reason="tantan is not installed"
)
@pytest.mark.parametrize("use_custom_matrix", [False, True])
def test_protein(use_custom_matrix):
    """
    Test masking a protein sequence based on a known example.
    """
    seq_string = "MAPKINASekinasekinase"
    sequence = seq.ProteinSequence(seq_string)

    if use_custom_matrix:
        matrix = align.SubstitutionMatrix.std_protein_matrix()
    else:
        matrix = None

    test_mask = TantanApp.mask_repeats(sequence, matrix)

    ref_mask = [True if char.islower() else False for char in seq_string]

    assert len(test_mask) == len(ref_mask)
    assert np.all(test_mask.tolist() == ref_mask)