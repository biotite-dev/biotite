# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import numpy as np

__all__ = ["find_subsequence", "find_symbol", "find_symbol_first",
           "find_symbol_last"]


def find_subsequence(sequence, query):
    if not sequence.get_alphabet().extends(query.get_alphabet()):
        raise ValueError("The sequences alphabets are not equal")
    match_indices = []
    frame_size = len(query)
    for i in range(len(sequence) - frame_size + 1):
        sub_seq_code = sequence.code[i : i + frame_size]
        if np.array_equal(query.code, sub_seq_code):
            match_indices.append(i)
    return np.array(match_indices)

def find_symbol(sequence, symbol):
    code = sequence.get_alphabet().encode(symbol)
    return np.where(sequence.code == code)[0]

def find_symbol_first(sequence, symbol):
    return np.min(find_symbol(sequence, symbol))
    
def find_symbol_last(sequence, symbol):
    return np.max(find_symbol(sequence, symbol))