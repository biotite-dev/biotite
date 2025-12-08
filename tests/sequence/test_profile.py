# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import numpy as np
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align


def test_from_alignment():
    ali_str = [
        "CGTCAT--",
        "--TCATGC",
    ]
    alignment = align.Alignment.from_strings(ali_str, seq.NucleotideSequence)
    profile = seq.SequenceProfile.from_alignment(alignment)
    symbols = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 2],
            [0, 2, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 2],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
        ]
    )
    gaps = np.array([1, 1, 0, 0, 0, 0, 1, 1])
    alphabet = seq.Alphabet(["A", "C", "G", "T"])
    assert np.array_equal(symbols, profile.symbols)
    assert np.array_equal(gaps, profile.gaps)
    assert alphabet == profile.alphabet


def test_to_consensus_nuc():
    symbols = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 2],
            [0, 2, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 2],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
        ]
    )
    gaps = np.array([1, 1, 0, 0, 0, 0, 1, 1])
    alphabet = seq.Alphabet(["A", "C", "G", "T"])
    profile = seq.SequenceProfile(symbols, gaps, alphabet)

    assert seq.NucleotideSequence("CGTCATGC") == profile.to_consensus()


def test_to_consensus_nuc_ambiguous():
    symbols = np.array(
        [
            [1, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 2],
            [0, 2, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 2],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
        ]
    )
    gaps = np.array([1, 1, 0, 0, 0, 0, 1, 1])
    alphabet = seq.Alphabet(["A", "C", "G", "T"])
    profile = seq.SequenceProfile(symbols, gaps, alphabet)

    assert seq.NucleotideSequence("MGTCATGC") == profile.to_consensus()


def test_to_consensus_prot():
    # Avidin protein sequence
    seq1 = seq.ProteinSequence(
        "MVHATSPLLLLLLLSLALVAPGLSARKCSLTGKWTNDLGSNMTIGAVNSRGEFTGTYITAVTATSNEIKESPLHGTQNTINKRTQP"
        "TFGFTVNWKFSESTTVFTGQCFIDRNGKEVLKTMWLLRSSVNDIGDDWKATRVGINIFTRLRTQKE"
    )
    # Streptavidin protein sequence
    seq2 = seq.ProteinSequence(
        "MRKIVVAAIAVSLTTVSITASASADPSKDSKAQVSAAEAGITGTWYNQLGSTFIVTAGADGALTGTYESAVGNAESRYVLTGRYDSA"
        "PATDGSGTALGWTVAWKNNYRNAHSATTWSGQYVGGAEARINTQWLLTSGTTEANAWKSTLVGHDTFTKVKPSAASIDAAKKAGVNN"
        "GNPLDAVQQ"
    )
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    alignment = align.align_optimal(seq1, seq2, matrix)[0]

    profile = seq.SequenceProfile.from_alignment(alignment)
    assert (
        seq.ProteinSequence(
            "MRHIATAAIALSLLLLSITALASADPGKDSKAQLSAAEAGITGKWTNDLGSNFIIGAVGADGAFTGTYESAVGNAESNEIKEGPLD"
            "GAPATDGKGTALGWTFAFKNNWKFAESATTFSGQCFGGADARINGKELLTKGTMEANAWKSTLLGHDSFSKVKDIAADIDAAKKAG"
            "INIFNPLDAQKE"
        )
        == profile.to_consensus()
    )


def test_new_position_matrices():
    seqs = [
        seq.NucleotideSequence("AAGAAT"),
        seq.NucleotideSequence("ATCATA"),
        seq.NucleotideSequence("AAGTAA"),
        seq.NucleotideSequence("AACAAA"),
        seq.NucleotideSequence("ATTAAA"),
        seq.NucleotideSequence("AAGAAT"),
    ]

    alignment = align.Alignment(
        sequences=seqs,
        trace=np.tile(np.arange(len(seqs[0])), len(seqs))
        .reshape(len(seqs), len(seqs[0]))
        .transpose(),
        score=0,
    )

    profile = seq.SequenceProfile.from_alignment(alignment)

    probability_matrix = np.array(
        [
            [
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            [0.66666667, 0.0, 0.0, 0.33333333],
            [0.0, 0.33333333, 0.5, 0.16666667],
            [0.83333333, 0.0, 0.0, 0.16666667],
            [0.83333333, 0.0, 0.0, 0.16666667],
            [0.66666667, 0.0, 0.0, 0.33333333],
        ]
    )

    ppm = profile.probability_matrix()

    assert np.allclose(probability_matrix, ppm, atol=1e-3)

    probability = profile.sequence_probability(seq.NucleotideSequence("AAAAAA"))

    assert probability == 0.0

    ppm = profile.probability_matrix(pseudocount=1)

    probability_matrix = np.array(
        [
            [0.89285714, 0.03571429, 0.03571429, 0.03571429],
            [0.60714286, 0.03571429, 0.03571429, 0.32142857],
            [0.03571429, 0.32142857, 0.46428571, 0.17857143],
            [0.75, 0.03571429, 0.03571429, 0.17857143],
            [0.75, 0.03571429, 0.03571429, 0.17857143],
            [0.60714286, 0.03571429, 0.03571429, 0.32142857],
        ]
    )

    assert np.allclose(probability_matrix, ppm, atol=1e-3)

    probability = profile.sequence_probability(
        seq.NucleotideSequence("AAAAAA"), pseudocount=1
    )

    assert probability == pytest.approx(0.0066, abs=1e-3)

    log_odds_matrix = np.array(
        [
            [1.83650127, -2.80735492, -2.80735492, -2.80735492],
            [1.28010792, -2.80735492, -2.80735492, 0.36257008],
            [-2.80735492, 0.36257008, 0.8930848, -0.48542683],
            [1.5849625, -2.80735492, -2.80735492, -0.48542683],
            [1.5849625, -2.80735492, -2.80735492, -0.48542683],
            [1.28010792, -2.80735492, -2.80735492, 0.36257008],
        ]
    )

    pwm = profile.log_odds_matrix(pseudocount=1)

    assert np.allclose(log_odds_matrix, pwm, atol=1e-3)

    score = profile.sequence_score(seq.NucleotideSequence("AAAAAA"), pseudocount=1)

    assert score == pytest.approx(4.7593, abs=1e-3)
