# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
import numpy as np
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align
import biotite.application.muscle as muscle
from ...util import is_not_installed
from .util import sequences


def test_align_ungapped():
    """
    Test `align_ungapped()` function via a known test case.
    """
    seq1 = seq.NucleotideSequence("ACCTGA")
    seq2 = seq.NucleotideSequence("ACTGGT")
    matrix = align.SubstitutionMatrix.std_nucleotide_matrix()
    ali = align.align_ungapped(seq1, seq2, matrix)
    assert ali.score == 3
    assert str(ali) == "ACCTGA\nACTGGT"


# [local, gap_penalty, input1, input2, expect]
align_cases = [(False,True, -7,      "TATGGGTATCC","TATGTATAA",
                    ("TATGGGTATCC\nTATG--TATAA",
                     "TATGGGTATCC\nTAT-G-TATAA",
                     "TATGGGTATCC\nTAT--GTATAA",)),
               (True, True, -6,      "TATGGGTATCC","TATGTATAA",
                    ("TATGGGTAT\nTATG--TAT",
                     "TATGGGTAT\nTAT-G-TAT",
                     "TATGGGTAT\nTAT--GTAT",)),
               (False,True, (-7,-1), "TACTATGGGTATCC","TCATATGTATAA",
                    ("TACTATGGGTATCC\nTCATATG--TATAA",
                     "TACTATGGGTATCC\nTCATAT--GTATAA",)),
               (True, True, (-7,-1), "TACTATGGGTATCC","TCATATGTATAA",
                    ("TATGGGTAT\nTATG--TAT",
                     "TATGGGTAT\nTAT--GTAT",)),
               (False,True, (-7,-1), "T","TTT",
                    ("T--\nTTT",
                     "--T\nTTT",)),
               (False,True, -7, "TAAAGCGAAAT","TGCGT",
                    ("TAAAGCGAAAT\nT---GCG---T")),
               (False,False,-7, "TAAAGCGAAAT","TGCGT",
                    ("TAAAGCGAAAT\n---TGCGT---"))
              ]
@pytest.mark.parametrize("local, term, gap_penalty, input1, input2, expect",
                         align_cases)
def test_align_optimal_simple(local, term, gap_penalty,
                              input1, input2, expect):
    """
    Test `align_optimal()` function using constructed test cases.
    """
    seq1 = seq.NucleotideSequence(input1)
    seq2 = seq.NucleotideSequence(input2)
    matrix = align.SubstitutionMatrix.std_nucleotide_matrix()
    # Test alignment function
    alignments = align.align_optimal(seq1, seq2,
                       matrix,
                       gap_penalty=gap_penalty, terminal_penalty=term,
                       local=local)
    
    for ali in alignments:
        assert str(ali) in expect
    # Test if separate score function calculates the same score
    for ali in alignments:
        score = align.score(ali, matrix,
                            gap_penalty=gap_penalty, terminal_penalty=term)
        assert score == ali.score


@pytest.mark.skipif(
    is_not_installed("muscle"),
    reason="MUSCLE is not installed"
)
# Ignore warning about MUSCLE writing no second guide tree
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("gap_penalty, seq_indices", itertools.product(
    [-10, (-10,-1)], [(i,j) for i in range(10) for j in range(i+1)]
))
def test_align_optimal_complex(sequences, gap_penalty, seq_indices):
    """
    Test `align_optimal()` function using real world sequences,
    compared to the output of MUSCLE.
    """
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    index1, index2 = seq_indices
    seq1 = sequences[index1]
    seq2 = sequences[index2]
    test_alignment = align.align_optimal(
        seq1, seq2, matrix,
        gap_penalty=gap_penalty, terminal_penalty=True, max_number=1
    )[0]
    ref_alignment = muscle.MuscleApp.align(
        [seq1, seq2], matrix=matrix, gap_penalty=gap_penalty
    )
    # Check whether the score of the optimal alignments is the same
    # or higher as the MUSCLE alignment
    # Direct alignment comparison is not feasible,
    # since the treatment of terminal gaps is different in MUSCLE
    test_score = align.score(
        test_alignment, matrix, gap_penalty, terminal_penalty=True
    )
    ref_score = align.score(
        ref_alignment, matrix, gap_penalty, terminal_penalty=True
    )
    try:
        assert test_score >= ref_score
    except AssertionError:
        print("Alignment:")
        print()
        print(test_alignment)
        print("\n")
        print("Reference alignment:")
        print()
        print(ref_alignment)
        raise


@pytest.mark.parametrize(
    "local, term, gap_penalty, seed", itertools.product(
        [True, False], [True, False], [-5, -8, -10, -15], range(10)
    )
)
def test_affine_gap_penalty(local, term, gap_penalty, seed):
    """
    Expect the same alignment results for a linear gap penalty and an
    affine gap penalty with the same gap open and extension penalty.
    """
    LENGTH_RANGE = (10, 100)
    MAX_NUMBER = 1000

    np.random.seed(seed)
    sequences = []
    for _ in range(2):
        sequence = seq.NucleotideSequence()
        length = np.random.randint(*LENGTH_RANGE)
        sequence.code = np.random.randint(
            len(sequence.alphabet), size=length
        )
        sequences.append(sequence)
    
    matrix = align.SubstitutionMatrix.std_nucleotide_matrix()

    ref_alignments = align.align_optimal(
        *sequences, matrix, gap_penalty, term, local, MAX_NUMBER
    )

    test_alignments = align.align_optimal(
        *sequences, matrix, (gap_penalty, gap_penalty), term, local, MAX_NUMBER
    )

    assert test_alignments[0].score == ref_alignments[0].score
    assert len(test_alignments) == len(ref_alignments)
    # We can only expect to get the same alignments in the test and
    # reference, if we get all optimal alignments
    if len(test_alignments) < MAX_NUMBER:
        for alignment in test_alignments:
            try:
                assert alignment in ref_alignments
            except:
                print("Test alignment:")
                print(alignment)
                print()
                print("First reference alignment")
                print(ref_alignments[0])
                raise


@pytest.mark.parametrize(
    "local, term, gap_penalty, seq_indices", itertools.product(
        [True, False], [True, False], [-10, (-10,-1)],
        [(i,j) for i in range(10) for j in range(i+1)]
    )
)
def test_align_optimal_symmetry(sequences, local, term, gap_penalty,
                                seq_indices):
    """
    Alignments should be indifferent about which sequence comes first.
    """
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    index1, index2 = seq_indices
    seq1 = sequences[index1]
    seq2 = sequences[index2]
    alignment1 = align.align_optimal(
        seq1, seq2, matrix,
        gap_penalty=gap_penalty, terminal_penalty=term, local=local,
        max_number=1
    )[0]
    # Swap the sequences
    alignment2 = align.align_optimal(
        seq2, seq1, matrix,
        gap_penalty=gap_penalty, terminal_penalty=term, local=local,
        max_number=1
    )[0]
    # Comparing all traces of both alignments to each other
    # would be unfeasible
    # Instead the scores are compared
    assert alignment1.score == alignment2.score


@pytest.mark.parametrize(
    "gap_penalty, term, seq_indices", itertools.product(
        [-10, (-10,-1)], [False, True],
        [(i,j) for i in range(10) for j in range(i+1)]
    )
)
def test_scoring(sequences, gap_penalty, term, seq_indices):
    """
    The result of the :func:`score()` function should be equal to the
    score calculated in :func:`align_optimal()`.
    """
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    index1, index2 = seq_indices
    seq1 = sequences[index1]
    seq2 = sequences[index2]
    alignment = align.align_optimal(
        seq1, seq2, matrix, gap_penalty=gap_penalty, terminal_penalty=term,
        max_number=1
    )[0]
    try:
        assert align.score(alignment, matrix, gap_penalty, term) \
               == alignment.score
    except AssertionError:
        print(alignment)
        raise