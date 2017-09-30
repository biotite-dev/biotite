# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import unittest
import biopython.sequence as seq
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
    matches = seq.find_symbol(dna, symbol)
    assert list(matches) == [1,6,7,10]