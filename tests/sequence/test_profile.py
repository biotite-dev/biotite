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


def test_new_position_matrices():
    seqs = [seq.NucleotideSequence("AAGAAT"),
            seq.NucleotideSequence("ATCATA"),
            seq.NucleotideSequence("AAGTAA"),
            seq.NucleotideSequence("AACAAA"),
            seq.NucleotideSequence("ATTAAA"),
            seq.NucleotideSequence("AAGAAT")]

    alignment = align.Alignment(
        sequences=seqs,
        trace=np.tile(np.arange(len(seqs[0])), len(seqs)) \
            .reshape(len(seqs), len(seqs[0])) \
            .transpose(),
        score=0
    )

    profile = seq.SequenceProfile.from_alignment(alignment)

    probability_matrix = np.array([[1., 0., 0., 0., ],
                                   [0.66666667, 0., 0., 0.33333333],
                                   [0., 0.33333333, 0.5, 0.16666667],
                                   [0.83333333, 0., 0., 0.16666667],
                                   [0.83333333, 0., 0., 0.16666667],
                                   [0.66666667, 0., 0., 0.33333333]])

    ppm = profile.probability_matrix()

    assert np.array_equal(np.around(probability_matrix, decimals=3), np.around(ppm, decimals=3))

    probability = profile.sequence_probability_from_matrix(seq.NucleotideSequence("AAAAAA"), ppm)

    assert probability == 0.0

    # Set pseudocount for position probability matrix to 1
    ppm = profile.probability_matrix(pseudocount=1)

    probability_matrix = np.array([[0.88095238, 0.02380952, 0.02380952, 0.02380952],
                                   [0.5952381, 0.02380952, 0.02380952, 0.30952381],
                                   [0.02380952, 0.30952381, 0.45238095, 0.16666667],
                                   [0.73809524, 0.02380952, 0.02380952, 0.16666667],
                                   [0.73809524, 0.02380952, 0.02380952, 0.16666667],
                                   [0.5952381, 0.02380952, 0.02380952, 0.30952381]])

    assert np.array_equal(np.around(probability_matrix, decimals=3), np.around(ppm, decimals=3))

    probability = profile.sequence_probability_from_matrix(seq.NucleotideSequence("AAAAAA"), ppm)

    assert np.around(probability, decimals=3) == np.around(0.004048642098725673, decimals=3)

    log_odds_matrix = np.array([[1.81713594, -3.39231742, -3.39231742, -3.39231742],
                                [1.25153877, -3.39231742, -3.39231742, 0.3081223],
                                [-3.39231742, 0.3081223, 0.85561009, -0.5849625],
                                [1.56187889, -3.39231742, -3.39231742, -0.5849625],
                                [1.56187889, -3.39231742, -3.39231742, -0.5849625],
                                [1.25153877, -3.39231742, -3.39231742, 0.3081223]])

    pwm = profile.log_odds_matrix(pseudocount=1)

    assert np.array_equal(np.around(log_odds_matrix, decimals=3), np.around(pwm, decimals=3))
