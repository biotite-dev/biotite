# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import unittest
import biopython.sequence as seq
import biopython.sequence.io.fasta as fasta
import biopython.database.entrez as entrez
import numpy as np
import os
import os.path
from .testutil import data_dir, uids

class FastaTest(unittest.TestCase):
    
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.files = entrez.fetch(uids, data_dir, "fasta", "nuccore", "fasta")
        
    def test_sequence_conversion(self):
        for path in self.files:
            with self.subTest(id=path[-8:-4]):
                fasta_file1 = fasta.FastaFile()
                fasta_file1.read(path)
                content1 = list(fasta_file1)
                fasta_file2 = fasta.FastaFile()
                for header, sequence in content1:
                    fasta_file2.add(header, sequence)
                content2 = list(fasta_file2)
                self.assertEqual(content1, content2)