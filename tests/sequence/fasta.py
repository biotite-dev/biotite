# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import unittest
import biopython.sequence as seq
import biopython.sequence.io.fasta as fasta
import numpy as np
import os
import os.path
from .testutil import data_dir

class FastaTest(unittest.TestCase):
    
    def test_access(self):
        path = os.path.join(data_dir, "nuc.fasta")
        file = fasta.FastaFile()
        file.read(path)
        self.assertEqual(file["dna sequence"], "ACGCTACGT")
        self.assertEqual(file["another dna sequence"], "A")
        self.assertEqual(file["third dna sequence"], "ACGT")
        self.assertEqual(dict(file),
                         {"dna sequence" : "ACGCTACGT",
                          "another dna sequence" : "A",
                          "third dna sequence" : "ACGT"})
        file["another dna sequence"] = "AA"
        del file["dna sequence"]
        file["yet another sequence"] = "ACGT"
        self.assertEqual(dict(file),
                         {"another dna sequence" : "AA",
                          "third dna sequence" : "ACGT",
                          "yet another sequence" : "ACGT"})
    
    def test_conversion(self):
        path = os.path.join(data_dir, "nuc.fasta")
        file = fasta.FastaFile()
        file.read(path)
        self.assertEqual(seq.NucleotideSequence("ACGCTACGT"),
                         fasta.get_sequence(file))
        
        seq_dict = fasta.get_sequences(file)
        file2 = fasta.FastaFile()
        fasta.set_sequences(file2, seq_dict)
        seq_dict2 = fasta.get_sequences(file2)
        self.assertEqual(seq_dict, seq_dict2)
        
        file3 = fasta.FastaFile()
        fasta.set_sequence(file3, seq.NucleotideSequence("AACCTTGG"))
        self.assertEqual(file3["sequence"], "AACCTTGG")
        
        path = os.path.join(data_dir, "prot.fasta")
        file4 = fasta.FastaFile()
        file4.read(path)
        self.assertEqual(seq.ProteinSequence("YAHGFRTGS"),
                         fasta.get_sequence(file4))
        
        path = os.path.join(data_dir, "invalid.fasta")
        file5 = fasta.FastaFile()
        file5.read(path)
        with self.assertRaises(ValueError):
            seq.NucleotideSequence(fasta.get_sequence(file5))
