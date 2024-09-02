# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import numpy as np
import pytest
import biotite.sequence as seq


def test_nucleotide_construction():
    string = "AATGCGTTA"
    string_amb = "ANNGCBRTAN"
    dna = seq.NucleotideSequence(string)
    assert dna.get_alphabet() == seq.NucleotideSequence.alphabet_unamb
    assert str(dna) == string
    dna = seq.NucleotideSequence(string_amb)
    assert dna.get_alphabet() == seq.NucleotideSequence.alphabet_amb
    assert str(dna) == string_amb


def test_reverse_complement():
    string = "AATGCGTTANY"
    dna = seq.NucleotideSequence(string)
    assert str(dna.reverse().complement()) == "RNTAACGCATT"


def test_stop_removal():
    string = "LYG*GR*"
    protein = seq.ProteinSequence(string)
    assert str(protein.remove_stops()) == string.replace("*", "")


@pytest.mark.parametrize(
    "dna_str, protein_str", [("CACATAGCATGA", "HIA*"), ("ATGTAGCTA", "M*L")]
)
def test_full_translation(dna_str, protein_str):
    dna = seq.NucleotideSequence(dna_str)
    protein = dna.translate(complete=True)
    assert protein_str == str(protein)


@pytest.mark.parametrize(
    "dna_str, protein_str_list",
    [
        ("CA", []),
        ("GAATGCACTGAGATGCAATAG", ["MH*", "MQ*"]),
        ("ATGCACATGTAGGG", ["MHM*", "M*"]),
        ("GATGCATGTGAAAA", ["MHVK", "M*"]),
    ],
)
def test_frame_translation(dna_str, protein_str_list):
    dna = seq.NucleotideSequence(dna_str)
    proteins, pos = dna.translate(complete=False)
    assert len(proteins) == len(protein_str_list)
    assert set([str(protein) for protein in proteins]) == set(protein_str_list)
    # Test if the positions are also right
    # -> Get sequence slice and translate completely
    assert set(
        [str(dna[start:stop].translate(complete=True)) for start, stop in pos]
    ) == set(protein_str_list)


def test_translation_met_start():
    """
    Test whether the start amino acid is replaced by methionine,
    i.e. the correct function of the 'met_start' parameter.
    """
    codon_table = seq.CodonTable.default_table().with_start_codons("AAA")
    dna = seq.NucleotideSequence("GAAACTGAAATAAGAAC")
    proteins, _ = dna.translate(codon_table=codon_table, met_start=True)
    assert [str(protein) for protein in proteins] == ["MLK*", "M*"]


def test_letter_conversion():
    for symbol in seq.ProteinSequence.alphabet:
        three_letters = seq.ProteinSequence.convert_letter_1to3(symbol)
        single_letter = seq.ProteinSequence.convert_letter_3to1(three_letters)
        assert symbol == single_letter


@pytest.mark.parametrize(
    "monoisotopic, expected_mol_weight_protein",
    # Reference values taken from https://web.expasy.org/compute_pi/
    [(True, 2231.06), (False, 2232.56)],
)
def test_get_molecular_weight(monoisotopic, expected_mol_weight_protein):
    """
    Test whether the molecular weight of a protein is calculated
    correctly.
    """
    protein = seq.ProteinSequence("ACDEFGHIKLMNPQRSTVW")
    mol_weight_protein = protein.get_molecular_weight(monoisotopic=monoisotopic)
    assert mol_weight_protein == pytest.approx(expected_mol_weight_protein, abs=1e-2)


def test_positional_sequence():
    """
    Check whether the original sequence symbols can be reconstructed from a
    :class:`PositionalSequence`.
    """
    SEQ_LENGTH = 100

    np.random.seed(0)
    orig_sequence = seq.ProteinSequence()
    orig_sequence.code = np.random.randint(
        0, len(orig_sequence.alphabet), size=SEQ_LENGTH
    )
    positional_sequence = seq.PositionalSequence(orig_sequence)

    assert str(positional_sequence) == str(orig_sequence)
    reconstructed_sequence = positional_sequence.reconstruct()
    assert np.all(reconstructed_sequence.code == orig_sequence.code)
    assert reconstructed_sequence.alphabet == orig_sequence.alphabet
