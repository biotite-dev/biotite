# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import unittest
import biotite.sequence as seq
import numpy as np
import pytest


def test_find_subsequence():
    string = "ATACGCTTGCT"
    substring = "GCT"
    main_seq = seq.NucleotideSequence(string)
    sub_seq = seq.NucleotideSequence(substring)
    matches = seq.find_subsequence(main_seq, sub_seq)
    assert list(matches) == [4,8]
    
def test_find_symbol():
    string = "ATACGCTTGCT"
    symbol = "T"
    dna = seq.NucleotideSequence(string)
    assert list(seq.find_symbol(dna, symbol)) == [1,6,7,10]
    assert seq.find_symbol_first(dna, symbol) == 1
    assert seq.find_symbol_last(dna, symbol) == 10