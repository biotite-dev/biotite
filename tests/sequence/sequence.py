# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import unittest
import biopython.sequence as seq
import numpy as np

class SequenceTest(unittest.TestCase):
    
    def test_encoding(self):
        string1 = "AATGCGTTA"
        dna = seq.NucleotideSequence(string1)
        string2 = str(dna)
        self.assertEqual(string1, string2)
        
    def test_access(self):
        string = "AATGCGTTA"
        dna = seq.NucleotideSequence(string)
        self.assertEqual(string[2], dna[2])
        dna[-1] = "C"
        self.assertEqual("AATGCGTTC", str(dna))
        dna = dna[3:-2]
        self.assertEqual("GCGT", str(dna))
        
    def test_alph_error(self):
        string = "AATGCGTUTA"
        with self.assertRaises(seq.AlphabetError):
            seq.NucleotideSequence(string)
    
    def test_reverse_complement(self):
        string = "AATGCGTTA"
        dna = seq.NucleotideSequence(string)
        self.assertEqual("TAACGCATT", str(dna.reverse().complement()))
        
    def test_translation(self):
        string = "AATGCGTTAGAT"
        dna = seq.NucleotideSequence(string)
        protein = dna.translate(complete=True)
        self.assertEqual("NALD", str(protein))
        proteins, pos = dna.translate(complete=False)
        self.assertEqual("MR*", str(proteins[0]))