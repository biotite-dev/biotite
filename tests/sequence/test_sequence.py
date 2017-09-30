# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import unittest
import biopython.sequence as seq
import numpy as np
import pytest


def test_encoding():
    string1 = "AATGCGTTA"
    dna = seq.NucleotideSequence(string1)
    string2 = str(dna)
    assert string1 == string2
    
def test_access():
    string = "AATGCGTTA"
    dna = seq.NucleotideSequence(string)
    assert string[2] == dna[2]
    dna[-1] = "C"
    assert "AATGCGTTC" == str(dna)
    dna = dna[3:-2]
    assert "GCGT" == str(dna)
    
def test_alph_error():
    string = "AATGCGTUTA"
    with pytest.raises(seq.AlphabetError):
        seq.NucleotideSequence(string)

def test_reverse_complement():
    string = "AATGCGTTA"
    dna = seq.NucleotideSequence(string)
    assert "TAACGCATT" == str(dna.reverse().complement())
    
def test_translation():
    string = "AATGCGTTAGAT"
    dna = seq.NucleotideSequence(string)
    protein = dna.translate(complete=True)
    assert "NALD" == str(protein)
    proteins, pos = dna.translate(complete=False)
    assert "MR*" == str(proteins[0])