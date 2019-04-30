# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
import itertools
import shutil
import numpy as np
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align
import biotite.sequence.io.fasta as fasta
import biotite.application.muscle as muscle
import biotite
from .util import data_dir


@pytest.fixture
def sequences():
    """
    10 Cas9 sequences.
    """
    fasta_file = fasta.FastaFile()
    fasta_file.read(join(data_dir, "cas9.fasta"))
    return [seq.ProteinSequence(sequence) for sequence in fasta_file.values()]

def test_alignment_str():
    """
    Test reading alignments from string.
    """
    seq1 = seq.NucleotideSequence("ACCTGA")
    seq2 = seq.NucleotideSequence("TATGCT")
    ali_str = ["A-CCTGA----",
               "----T-ATGCT"]
    trace = align.Alignment.trace_from_strings(ali_str)
    alignment = align.Alignment([seq1, seq2], trace, None)
    assert str(alignment).split("\n") == ali_str

def test_conversion_to_symbols():
    """
    Test conversion of alignments to strings.
    """
    seq_str1 = "HAKLPRDD--WKL--"
    seq_str2 = "HA--PRDDADWKLHH"
    seq_str3 = "HA----DDADWKLHH"
    seq_strings = [seq_str1, seq_str2, seq_str3]
    sequences = [seq.ProteinSequence(seq_str.replace("-",""))
                 for seq_str in seq_strings]
    trace = align.Alignment.trace_from_strings(seq_strings)
    alignment = align.Alignment(sequences, trace, score=None)
    # Test the conversion bach to strings of symbols
    symbols = align.get_symbols(alignment)
    symbols = ["".join([sym if sym is not None else "-" for sym in sym_list])
               for sym_list in symbols]
    assert symbols == seq_strings

def test_identity():
    """
    Test correct calculation of sequence identity.
    """
    seq_str1 = "--HAKLPRDD--WL--"
    seq_str2 = "FRHA--QRTDADWLHH"
    seq_strings = [seq_str1, seq_str2]
    sequences = [seq.ProteinSequence(seq_str.replace("-",""))
                 for seq_str in seq_strings]
    trace = align.Alignment.trace_from_strings(seq_strings)
    alignment = align.Alignment(sequences, trace, score=None)
    # Assert correct sequence identity calculation
    modes = ["all", "not_terminal", "shortest"]
    values = [6/16, 6/12, 6/10]
    for mode, value in zip(modes, values):
        assert align.get_sequence_identity(alignment, mode=mode) == value

def test_align_ungapped():
    """
    Test `align_ungapped()` function.
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
    shutil.which("muscle") is None,
    reason="MUSCLE is not installed"
)
@pytest.mark.parametrize("gap_penalty, seq_indices", itertools.product(
    [-10, (-10,-1)], [(i,j) for i in range(10) for j in range(i+1)]
))
def test_align_optimal_complex(sequences, gap_penalty, seq_indices):
    """
    Test `align_optimal()` function using actual long sequences,
    compared to the output of MUSCLE.
    """
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    index1, index2 = seq_indices
    seq1 = sequences[index1]
    seq2 = sequences[index2]
    alignment = align.align_optimal(
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
    score = align.score(alignment, matrix, gap_penalty, terminal_penalty=True)
    ref_score = align.score(
        ref_alignment, matrix, gap_penalty, terminal_penalty=True
    )
    try:
        assert score >= ref_score
    except AssertionError:
        print("Alignment:")
        print()
        print(alignment)
        print("\n")
        print("Reference alignment:")
        print()
        print(alignment)
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
    Test `score()` function.
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

@pytest.mark.skipif(
    shutil.which("muscle") is None,
    reason="MUSCLE is not installed"
)
@pytest.mark.parametrize("gap_penalty", [-10, (-10,-1)])
def test_align_multiple(sequences, gap_penalty):
    r"""
    Test `align_multiple()` function using actual long sequences,
    compared to the output of MUSCLE.
    Both alignment methods are heuristic, the exact same result is not
    expected.
    Just assert that the resulting score is at least the 50 % of the
    score of the MUSCLE alignment.
    """
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    alignment, order, tree, distances = align.align_multiple(
        sequences, matrix, gap_penalty=gap_penalty, terminal_penalty=True
    )
    ref_alignment = muscle.MuscleApp.align(
        sequences, matrix=matrix, gap_penalty=gap_penalty
    )
    score = align.score(alignment, matrix, gap_penalty, terminal_penalty=True)
    ref_score = align.score(
        ref_alignment, matrix, gap_penalty, terminal_penalty=True
    )
    assert score >= ref_score * 0.5

@pytest.mark.parametrize("db_entry", [entry for entry
                                      in align.SubstitutionMatrix.list_db()
                                      if entry not in ["NUC","GONNET"]])
def test_matrices(db_entry):
    """
    Test reading of matrix files.
    """
    alph1 = seq.ProteinSequence.alphabet
    alph2 = seq.ProteinSequence.alphabet
    matrix = align.SubstitutionMatrix(alph1, alph2, db_entry)

def test_matrix_str():
    """
    Test conversion of substitution matrix to string.
    """
    alph1 = seq.Alphabet("abc")
    alph2 = seq.Alphabet("def")
    score_matrix = np.arange(9).reshape((3,3))
    matrix = align.SubstitutionMatrix(alph1, alph2, score_matrix)
    assert str(matrix) == "\n".join(
        ["    d   e   f",
         "a   0   1   2",
         "b   3   4   5",
         "c   6   7   8"]
    )
