# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import unittest
import biopython.sequence as seq
import numpy as np

class SearchTest(unittest.TestCase):
    
    def test_find_subsequence(self):
        string = "ATACGCTTGCT"
        substring = "GCT"
        main_seq = seq.DNASequence(string)
        sub_seq = seq.DNASequence(substring)
        matches = seq.find_subsequence(main_seq, sub_seq)
        self.assertListEqual(list(matches), [4,8])
        
    def test_find_symbol(self):
        string = "ATACGCTTGCT"
        symbol = "T"
        dna = seq.DNASequence(string)
        matches = seq.find_symbol(dna, symbol)
        self.assertListEqual(list(matches), [1,6,7,10])