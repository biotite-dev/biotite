# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import unittest
import biopython.sequence as seq
import biopython.application.blast as blast
import numpy as np

class BlastTest(unittest.TestCase):
    
    def test_online_blastp(self):
        prot_seq = seq.ProteinSequence("NLYIQWLKDGGPSSGRPPPS")
        app = blast.BlastOnline("blastp", prot_seq)
        app.start()
        app.join()
        alignments = app.get_alignments()
        self.assertEqual(prot_seq, alignments[0].seq1)
        self.assertEqual(prot_seq, alignments[0].seq2)