# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

from ..sequence import Sequence

def simple_score(seq1, seq2, matrix):
    if len(seq1) != len(seq2):
        raise ValueError("Sequence lengths of {:d} and {:d} are not equal"
                         .format( len(seq1), len(seq2) ))
    if (matrix.alphabet1() != seq1.get_alphabet() and
        matrix.alphabet2() != seq2.get_alphabet()):
            raise ValueError("The sequences' alphabets do not fit the matrix")
    seq1_code = seq1.get_seq_code()
    seq2_code = seq1.get_seq_code()
    score = 0
    for i in range(len(seq1)):
        score += matrix.get_score_by_code(seq1_code[i], seq2_code[i])
    return code