# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import numpy as np


def find_subsequence(sequence, query):
    if not sequence.get_alphabet.extends(query.get_alphabet()):
        raise ValueError("The sequences alphabets are not equal")
    match_indices = []
    frame_size = len(query)
    for i in range(len(sequence) - frame_size + 1):
        sub_seq = sequence._seq_code[i : i + frame_size]
        if np.array_equal(query, sub_seq):
            match_indices.append(i)
    return match_indices

def find_symbol(sequence, symbol):
    code = sequence.get_alphabet().encode(symbol)
    return numpy.where(self._seq_code == code)