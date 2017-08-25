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
        seq1 = seq.DNASequence("ATACGCTTGCT")
        seq2 = seq.DNASequence("AGGCGCAGCT")
        alignments = align.align_global(seq1, seq2,
                           align.SubstitutionMatrix.std_nucleotide_matrix(),
                           gap_opening=-6,
                           gap_extension=-2)
        alignment = alignments[0]
        self.assertEqual(alignment.trace.tolist(),
                         [[0,0], [1,1], [2,2], [3,3], [4,4], [5,5],
                          [6,6], [7,-1], [8,7], [9,8], [10,9]])
    
    def test_local_align(self):
        seq1 = seq.DNASequence("ATGGCACATGATCTTA")
        seq2 = seq.DNASequence("ACTTGCTTACGAT")
        alignments = align.align_local(seq1, seq2,
                           align.SubstitutionMatrix.std_nucleotide_matrix(),
                           gap_opening=-6,
                           gap_extension=-2)
        alignment = alignments[0]
        self.assertEqual(alignment.trace.tolist(),
                         [[5,0], [6,1], [7,2], [8,3], [9,4], [10,-1],
                          [11,-1], [12,5], [13,6], [14,7], [15,8]])