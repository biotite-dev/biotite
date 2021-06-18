# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
import pytest
import numpy as np
import biotite.sequence as seq
import biotite.sequence.align as align
from .util import sequences


@pytest.mark.parametrize(
    "gap_penalty, seed, threshold, direction, score_only",
    itertools.product(
        [-10, (-10,-1)],
        [(0, 0), (11, 11), (20, 19), (30, 29)],
        [20, 100, 500],
        ["both", "upstream","downstream"],
        [False, True]
    )
)
def test_simple_alignment(gap_penalty, seed, threshold,
                          direction, score_only):
    """
    Test `align_local_gapped()` by comparing the output to
    `align_optimal()`.
    This test uses a pair of highly similar short sequences.
    """
    # Cyclotide C, Uniprot: P86843
    seq1 = seq.ProteinSequence("gvpcaescvwipctvtallgcsckdkvcyld")
    # Cyclotide F, Uniprot: P86846
    seq2 = seq.ProteinSequence("gipcgescvfipcissvvgcsckskvcyld")
    matrix = align.SubstitutionMatrix.std_protein_matrix()

    ref_alignments = align.align_optimal(
        seq1, seq2, matrix,
        gap_penalty=gap_penalty, local=True
    )
    # Limit reference alignment range to seed
    # if the alignment does not extend in both directions
    for alignment in ref_alignments:
        seed_index = np.where(alignment.trace[:,0] == seed[0])[0][0]
        if direction == "upstream":
            alignment.trace = alignment.trace[:seed_index + 1]
        elif direction == "downstream":
            alignment.trace = alignment.trace[seed_index:]
        alignment.score = align.score(alignment, matrix, gap_penalty)
    
    test_result = align.align_local_gapped(
        seq1, seq2, matrix, seed, threshold, gap_penalty,
        1000, direction, score_only
    )

    if score_only:
        test_score = test_result
        # All optimal alignments have the same score
        assert test_score == ref_alignments[0].score
    else:
        test_alignments = test_result
        assert len(test_alignments) == len(ref_alignments)
        for alignment in test_alignments:
            assert alignment in ref_alignments


@pytest.mark.parametrize(
    "gap_penalty, score_only, seq_indices",
    itertools.product(
        [-10, (-10,-1)],
        [False, True],
        [(i,j) for i in range(10) for j in range(i+1)]
        #[(0,1)]
    )
)
def test_complex_alignment(sequences, gap_penalty, score_only,
                           seq_indices):
    """
    Test `align_local_gapped()` by comparing the output to
    `align_optimal()`.
    This test uses a set of long sequences, which are pairwise compared.
    The threshold should be chosen sufficiently large so
    `align_local_gapped()` can return the optimal alignment(s).
    """
    MAX_NUMBER = 100
    # The linear gap penalty for longer gaps easily exceeds
    # a small threshold -> increase threshold for linear penalty
    THRESHOLD = 200 if isinstance(gap_penalty, int) else 50
    
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    index1, index2 = seq_indices
    seq1 = sequences[index1]
    seq2 = sequences[index2]

    ref_alignments = align.align_optimal(
        seq1, seq2, matrix,
        gap_penalty=gap_penalty, local=True, max_number=MAX_NUMBER
    )
    # Select the center of the alignment as seed
    trace = ref_alignments[0].trace
    trace = trace[(trace != -1).all(axis=1)]
    seed = trace[len(trace) // 2]
    
    test_result = align.align_local_gapped(
        seq1, seq2, matrix, seed, THRESHOLD, gap_penalty,
        MAX_NUMBER, "both", score_only
    )

    if score_only:
        test_score = test_result
        # All optimal alignments have the same score
        assert test_score == ref_alignments[0].score
    else:
        try:
            test_alignments = test_result
            assert test_alignments[0].score == ref_alignments[0].score
            # Test if the score is also correctly calculated
            assert align.score(test_alignments[0], matrix, gap_penalty) \
                == ref_alignments[0].score
            if len(ref_alignments) < MAX_NUMBER \
               and len(test_alignments) < MAX_NUMBER:
                    # Only test if the exact same alignments were created,
                    # if the number of traces was not limited by MAX_NUMBER
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