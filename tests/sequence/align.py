# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import unittest
import biopython.sequence as seq
import biopython.sequence.align as align
import numpy as np

class AlignTest(unittest.TestCase):
    
    def test_align_global(self):
        seq1 = seq.NucleotideSequence("ATACGCTTGCT")
        seq2 = seq.NucleotideSequence("AGGCGCAGCT")
        alignments = align.align_global(seq1, seq2,
                           align.SubstitutionMatrix.std_nucleotide_matrix(),
                           gap_opening=-6,
                           gap_extension=-2)
        alignment = alignments[0]
        self.assertEqual(alignment.trace.tolist(),
                         [[0,0], [1,1], [2,2], [3,3], [4,4], [5,5],
                          [6,6], [7,-1], [8,7], [9,8], [10,9]])
    
    def test_align_local(self):
        seq1 = seq.NucleotideSequence("ATGGCACATGATCTTA")
        seq2 = seq.NucleotideSequence("ACTTGCTTACGAT")
        alignments = align.align_local(seq1, seq2,
                           align.SubstitutionMatrix.std_nucleotide_matrix(),
                           gap_opening=-6,
                           gap_extension=-2)
        alignment = alignments[0]
        self.assertEqual(alignment.trace.tolist(),
                         [[5,0], [6,1], [7,2], [8,3], [9,4], [10,-1],
                          [11,-1], [12,5], [13,6], [14,7], [15,8]])
    
    def test_matrices(self):
        db_entries = align.SubstitutionMatrix.list_db()
        for entry in db_entries:
            with self.subTest(entry):
                if entry not in ["NUC","GONNET"]:
                    alph1 = seq.ProteinSequence.alphabet
                    alph2 = seq.ProteinSequence.alphabet
                    matrix = align.SubstitutionMatrix(alph1, alph2, entry)