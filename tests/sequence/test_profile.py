# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import numpy as np
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align


def test_from_alignment():
    seq1 = seq.NucleotideSequence("CGTCAT")
    seq2 = seq.NucleotideSequence("TCATGC")
    ali_str = ["CGTCAT--",
               "--TCATGC"]
    trace = align.Alignment.trace_from_strings(ali_str)
    alignment = align.Alignment([seq1, seq2], trace, None)

    profile = seq.SequenceProfile.from_alignment(alignment)
    symbols = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 2], [0, 2, 0, 0],
                        [2, 0, 0, 0], [0, 0, 0, 2], [0, 0, 1, 0], [0, 1, 0, 0]])
    gaps = np.array([1, 1, 0, 0, 0, 0, 1, 1])
    alphabet = seq.Alphabet(["A", "C", "G", "T"])
    assert np.array_equal(symbols, profile.symbols)
    assert np.array_equal(gaps, profile.gaps)
    assert (alphabet == profile.alphabet)


def test_to_consensus_nuc():
    symbols = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 2], [0, 2, 0, 0],
                        [2, 0, 0, 0], [0, 0, 0, 2], [0, 0, 1, 0], [0, 1, 0, 0]])
    gaps = np.array([1, 1, 0, 0, 0, 0, 1, 1])
    alphabet = seq.Alphabet(["A", "C", "G", "T"])
    profile = seq.SequenceProfile(symbols, gaps, alphabet)

    assert seq.NucleotideSequence("CGTCATGC") == profile.to_consensus()


def test_to_consensus_nuc_ambiguous():
    symbols = np.array([[1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 2], [0, 2, 0, 0],
                        [2, 0, 0, 0], [0, 0, 0, 2], [0, 0, 1, 0], [0, 1, 0, 0]])
    gaps = np.array([1, 1, 0, 0, 0, 0, 1, 1])
    alphabet = seq.Alphabet(["A", "C", "G", "T"])
    profile = seq.SequenceProfile(symbols, gaps, alphabet)

    assert seq.NucleotideSequence("MGTCATGC") == profile.to_consensus()


def test_to_consensus_prot():
    # Avidin protein sequence
    seq1 = seq.ProteinSequence("MVHATSPLLLLLLLSLALVAPGLSARKCSLTGKWTNDLGSNMTIGAVNSRGEFTGTYITAVTATSNEIKESPLHGTQNTINKRTQP"
                               "TFGFTVNWKFSESTTVFTGQCFIDRNGKEVLKTMWLLRSSVNDIGDDWKATRVGINIFTRLRTQKE")
    # Streptavidin protein sequence
    seq2 = seq.ProteinSequence("MRKIVVAAIAVSLTTVSITASASADPSKDSKAQVSAAEAGITGTWYNQLGSTFIVTAGADGALTGTYESAVGNAESRYVLTGRYDSA"
                               "PATDGSGTALGWTVAWKNNYRNAHSATTWSGQYVGGAEARINTQWLLTSGTTEANAWKSTLVGHDTFTKVKPSAASIDAAKKAGVNN"
                               "GNPLDAVQQ")
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    alignment = align.align_optimal(seq1, seq2, matrix)[0]

    profile = seq.SequenceProfile.from_alignment(alignment)
    assert seq.ProteinSequence("MRHIATAAIALSLLLLSITALASADPGKDSKAQLSAAEAGITGKWTNDLGSNFIIGAVGADGAFTGTYESAVGNAESNEIKEGPLD"
                               "GAPATDGKGTALGWTFAFKNNWKFAESATTFSGQCFGGADARINGKELLTKGTMEANAWKSTLLGHDSFSKVKDIAADIDAAKKAG"
                               "INIFNPLDAQKE") == profile.to_consensus()
