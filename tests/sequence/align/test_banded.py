# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align
from .util import sequences


@pytest.mark.parametrize(
    "gap_penalty, local, band_width", itertools.product(
    [-10, ],#(-10,-1)],
    [False, True],
    [2, 5, 20, 100]
))
def test_align_banded_simple(gap_penalty, local, band_width):
    # Cyclotide C, Uniprot: P86843
    seq1 = seq.ProteinSequence("gvpcaescvwipctvtallgcsckdkvcyld")
    # Cyclotide F, Uniprot: P86846
    seq2 = seq.ProteinSequence("gipcgescvfipcissvvgcsckskvcyld")
    matrix = align.SubstitutionMatrix.std_protein_matrix()

    ref_alignments = align.align_optimal(
        seq1, seq2, matrix,
        gap_penalty=gap_penalty, local=local, terminal_penalty=False
    )
    # Remove terminal gaps in reference to obtain a true semi-global
    # alignment, as returned by align_banded()
    ref_alignments = [align.remove_terminal_gaps(al) for al in ref_alignments]
    
    test_alignments = align.align_banded(
        seq1, seq2, matrix, (-band_width, band_width)
    )

    assert len(test_alignments) == len(ref_alignments)
    for alignment in test_alignments:
        assert alignment in ref_alignments


@pytest.mark.parametrize(
    "gap_penalty, local, seq_indices", itertools.product(
    [-10, ],#(-10,-1)],
    [False, True],
    [(i,j) for i in range(10) for j in range(i+1)]
))
def test_align_banded_complex(sequences, gap_penalty, local, seq_indices):
    """
    Test `align_banded()` using real world sequences,
    compared to the output of `align_optimal()`.
    The band should be chosen sufficiently large so `align_banded()`
    can return the optimal alignment(s).
    """
    MAX_NUMBER = 100
    
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    index1, index2 = seq_indices
    seq1 = sequences[index1]
    seq2 = sequences[index2]

    ref_alignments = align.align_optimal(
        seq1, seq2, matrix,
        gap_penalty=gap_penalty, local=local, terminal_penalty=False,
        max_number=MAX_NUMBER
    )
    # Remove terminal gaps in reference to obtain a true semi-global
    # alignment, as returned by align_banded()
    ref_alignments = [align.remove_terminal_gaps(al) for al in ref_alignments]
    
    identity = align.get_sequence_identity(ref_alignments[0])
    # Use a relatively small band width, if the sequences are similar,
    # otherwise use the entire search space
    band_width = 100 if identity > 0.5 else len(seq1) + len(seq2)
    test_alignments = align.align_banded(
        seq1, seq2, matrix, (-band_width, band_width),
        gap_penalty=gap_penalty, local=local, max_number=MAX_NUMBER
    )

    try:
        assert test_alignments[0].score == ref_alignments[0].score
        if len(ref_alignments) < MAX_NUMBER:
            # Only test if the exact same alignments were created,
            # if the number of traces was not limited by MAX_NUMBER
            assert len(test_alignments) == len(ref_alignments)
            for alignment in test_alignments:
                assert alignment in ref_alignments
    except AssertionError:
        print("First tested alignment:")
        print()
        print(test_alignments[0])
        print("\n")
        print("First reference alignment:")
        print()
        print(ref_alignments[0])
        raise


def test_swapping():
    pass