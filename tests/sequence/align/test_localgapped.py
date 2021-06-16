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
    if gap_penalty == (-10,-1):
        pytest.skip()
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
    if gap_penalty == (-10,-1):
        pytest.skip()
    MAX_NUMBER = 100
    THRESHOLD = 1000
    
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


#@pytest.mark.parametrize(
#    "gap_penalty, local, seed", itertools.product(
#    [-10, (-10, -1)],
#    [False, True],
#    range(100)
#))
#def test_swapping(gap_penalty, local, seed):
#    """
#    Check if `align_banded()` returns a 'swapped' alignment, if
#    the order of input sequences is swapped.
#    """
#    np.random.seed(seed)
#    band = (
#        np.random.randint(-30, -10),
#        np.random.randint( 10,  30)
#    )
#
#    seq1, seq2 = _create_random_pair(seed)
#    matrix = align.SubstitutionMatrix.std_protein_matrix()
#    ref_alignments = align.align_banded(
#        seq1, seq2, matrix, band=band, local=local, gap_penalty=gap_penalty
#    )
#
#    test_alignments = align.align_banded(
#        seq2, seq1, matrix, band=band, local=local, gap_penalty=gap_penalty
#    )
#
#    if len(ref_alignments) != 1 or  len(test_alignments) != 1:
#        # If multiple optimal alignments exist,
#        # it is not easy to assign a swapped one to an original one
#        # therefore, simply return in this case
#        # the number of tested seeds should be large enough to generate
#        # a reasonable number of suitable test cases
#        return
#    ref_alignment = ref_alignments[0]
#    test_alignment = test_alignments[0]
#    
#    assert test_alignment.sequences[0] == ref_alignment.sequences[1]
#    assert test_alignment.sequences[1] == ref_alignment.sequences[0]
#    assert np.array_equal(test_alignment.trace, ref_alignment.trace[:, ::-1])
#
#
#
#def _create_random_pair(seed, length=100, max_subsitutions=5,
#                        max_insertions=5, max_deletions=5,
#                        max_truncations=5):
#    """
#    generate a pair of protein sequences.
#    Each pair contains
#
#        1. a randomly generated sequence
#        2. a sequence created by randomly introducing
#           deletions/insertions/substitutions into the first sequence.
#    """
#    np.random.seed(seed)
#
#    original = seq.ProteinSequence()
#    original.code = np.random.randint(len(original.alphabet), size=length)
#
#    mutant = original.copy()
#
#    # Random Substitutions
#    n_subsitutions = np.random.randint(max_subsitutions)
#    subsitution_indices = np.random.choice(
#        np.arange(len(mutant)), size=n_subsitutions, replace=False
#    )
#    subsitution_values = np.random.randint(
#        len(original.alphabet), size=n_subsitutions
#    )
#    mutant.code[subsitution_indices] = subsitution_values
#
#    # Random insertions
#    n_insertions = np.random.randint(max_insertions)
#    insertion_indices = np.random.choice(
#        np.arange(len(mutant)), size=n_insertions, replace=False
#    )
#    insertion_values = np.random.randint(
#        len(original.alphabet), size=n_insertions
#    )
#    mutant.code = np.insert(mutant.code, insertion_indices, insertion_values)
#
#    # Random deletions
#    n_deletions = np.random.randint(max_deletions)
#    deletion_indices = np.random.choice(
#        np.arange(len(mutant)), size=n_deletions, replace=False
#    )
#    mutant.code = np.delete(mutant.code, deletion_indices)
#
#    # Truncate at both ends of original and mutant
#    original = original[
#        np.random.randint(max_truncations) :
#        -(1 + np.random.randint(max_truncations))
#    ]
#    mutant = mutant[
#        np.random.randint(max_truncations) :
#        -(1 + np.random.randint(max_truncations))
#    ]
#
#    return original, mutant