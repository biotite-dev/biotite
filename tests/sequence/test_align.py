# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite.sequence as seq
import biotite.sequence.align as align
import biotite
import numpy as np
import pytest


def test_alignment_str():
    seq1 = seq.NucleotideSequence("ACCTGA")
    seq2 = seq.NucleotideSequence("TATGCT")
    ali_str = ["A-CCTGA----",
               "----T-ATGCT"]
    trace = align.Alignment.trace_from_strings(ali_str)
    alignment = align.Alignment([seq1, seq2], trace, None)
    assert str(alignment).split("\n") == ali_str

def test_simple_score():
    seq1 = seq.NucleotideSequence("ACCTGA")
    seq2 = seq.NucleotideSequence("ACTGGT")
    matrix = align.SubstitutionMatrix.std_nucleotide_matrix()
    assert align.simple_score(seq1, seq2, matrix) == 3

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
def test_align_optimal(local, term, gap_penalty,
                       input1, input2, expect):
    seq1 = seq.NucleotideSequence(input1)
    seq2 = seq.NucleotideSequence(input2)
    alignments = align.align_optimal(seq1, seq2,
                       align.SubstitutionMatrix.std_nucleotide_matrix(),
                       gap_penalty=gap_penalty, terminal_penalty=term,
                       local=local)
    for ali in alignments:
        assert str(ali) in expect

@pytest.mark.parametrize("db_entry", [entry for entry
                                      in align.SubstitutionMatrix.list_db()
                                      if entry not in ["NUC","GONNET"]])
def test_matrices(db_entry):
    alph1 = seq.ProteinSequence.alphabet
    alph2 = seq.ProteinSequence.alphabet
    matrix = align.SubstitutionMatrix(alph1, alph2, db_entry)

def test_matrix_str():
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
