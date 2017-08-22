# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import unittest
import biopython.sequence as seq
import biopython.sequence.align as align
import numpy as np

class AlignTest(unittest.TestCase):
    
    def test_global_align(self):
        string1 = "ATACGCTTGCT"
        string2 = "AGGCGCAGCT"
        seq1 = seq.DNASequence(string1)
        seq2 = seq.DNASequence(string2)
        alignment = align.align_global(seq1, seq2,
                           align.SubstitutionMatrix.std_nucleotide_matrix(),
                           gap_opening=-6,
                           gap_extension=-2)
        alignment = alignment[0]
        self.assertEqual(alignment.trace.tolist(),
                         [[0,0], [1,1], [2,2], [3,3], [4,4], [5,5],
                          [6,6], [7,-1], [8,7], [9,8], [10,9]])