# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

import unittest
import biopython.sequence as seq
import numpy as np
import pytest


def test_nucleotide_construction():
    string = "AATGCGTTA"
    string_amb = "ANNGCBRTAX"
    dna = seq.NucleotideSequence(string)
    assert dna.get_alphabet() == seq.NucleotideSequence.alphabet
    assert str(dna) == string
    dna = seq.NucleotideSequence(string_amb)
    assert dna.get_alphabet() == seq.NucleotideSequence.alphabet_amb
    assert str(dna) == string_amb

def test_reverse_complement():
    string = "AATGCGTTA"
    dna = seq.NucleotideSequence(string)
    assert "TAACGCATT" == str(dna.reverse().complement())

def test_stop_removal():
    string = "LYG*GR*"
    protein = seq.ProteinSequence(string)
    assert str(protein.remove_stops()) == string.replace("*", "")

@pytest.mark.parametrize("dna_str, protein_str",
                         [("CACATAGCATGA", "HIA*"),
                          ("ATGTAGCTA", "M*L")])
def test_full_translation(dna_str, protein_str):
    dna = seq.NucleotideSequence(dna_str)
    protein = dna.translate(complete=True)
    assert protein_str == str(protein)

@pytest.mark.parametrize("dna_str, protein_str_list",
                         [("CA", []),
                          ("GAATGCACTGAGATGCAATAG", ["MH*","MQ*"]),
                          ("ATGCACATGTAGGG", ["MHM*","M*"]),
                          ("GATGCATGTGAAAA", ["MHVK","M*"])])
def test_frame_translation(dna_str, protein_str_list):
    dna = seq.NucleotideSequence(dna_str)
    proteins, pos = dna.translate(complete=False)
    assert protein_str_list == [str(protein) for protein in proteins]

def test_letter_conversion():
    for symbol in seq.ProteinSequence.alphabet:
        three_letters = seq.ProteinSequence.convert_letter_1to3(symbol)
        single_letter = seq.ProteinSequence.convert_letter_3to1(three_letters)
        assert symbol == single_letter