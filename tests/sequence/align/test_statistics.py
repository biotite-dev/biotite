# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import pytest
import numpy as np
import biotite.sequence as seq
import biotite.sequence.align as align
from biotite.sequence.align.statistics import EValueEstimator
from .util import sequences


BACKGROUND = np.array(list({
    "A": 35155,
    "C": 8669,
    "D": 24161,
    "E": 28354,
    "F": 17367,
    "G": 33229,
    "H": 9906,
    "I": 23161,
    "K": 25872,
    "L": 40625,
    "M": 10101,
    "N": 20212,
    "P": 23435,
    "Q": 19208,
    "R": 23105,
    "S": 32070,
    "T": 26311,
    "V": 29012,
    "W": 5990,
    "Y": 14488,
    "B": 0,
    "Z": 0,
    "X": 0,
    "*": 0,
}.values())) / 450431


@pytest.mark.parametrize(
    "matrix_name, gap_penalty, ref_lam, ref_k",
    [
        ("BLOSUM62", (-10000, -10000), 0.318, 0.130),
        ("BLOSUM62", (   -12,     -2), 0.300, 0.090),
        ("BLOSUM62", (    -5,     -5), 0.131, 0.009),
        (  "PAM250", (   -16,     -1), 0.172, 0.018),
    ]
)
def test_distribution_param(matrix_name, gap_penalty, ref_lam, ref_k):
    """
    Check if `EValueEstimator` estimates the extreme value distribution
    parameters correctly by comparing them to the parameters described
    in the original publication by Altschul *et al*.
    """
    SAMPLE_LENGTH = 500
    SAMPLE_SIZE = 1000
    
    alphabet = seq.ProteinSequence.alphabet
    matrix = align.SubstitutionMatrix(alphabet, alphabet, matrix_name)

    np.random.seed(0)
    estimator = align.EValueEstimator.from_samples(
        alphabet, matrix, gap_penalty, BACKGROUND,
        SAMPLE_LENGTH, SAMPLE_SIZE
    )

    # Due to relatively low sample size, expect rather large deviation
    assert estimator.lam == pytest.approx(ref_lam, rel=0.1)
    assert estimator.k == pytest.approx(ref_k, rel=0.6)


def test_evalue():
    """
    Check if the estimated E-values for a given score approximately
    match the number of random sequences with equal or better score via
    sampling.
    Low scores that lead to a rather high E-value are required to get
    a reasonable accuracy.
    """
    TEST_SCORES = [30, 40, 50]
    GAP_PENALTY = (-12, -1)
    N_SAMPLES = 10000
    SEQ_LENGTH = 300

    matrix = align.SubstitutionMatrix.std_protein_matrix()
    estimator = align.EValueEstimator.from_samples(
        seq.ProteinSequence.alphabet, matrix, GAP_PENALTY,
        BACKGROUND
    )

    # Generate large number of alignments of random sequences
    np.random.seed(0)
    random_sequence_code = np.random.choice(
        len(seq.ProteinSequence.alphabet),
        size=(N_SAMPLES, 2, SEQ_LENGTH),
        p=BACKGROUND
    )
    sample_scores = np.zeros(N_SAMPLES, dtype=int)
    for i in range(N_SAMPLES):
        seq1 = seq.ProteinSequence()
        seq2 = seq.ProteinSequence()
        seq1.code = random_sequence_code[i,0]
        seq2.code = random_sequence_code[i,1]
        sample_scores[i] = align.align_optimal(
            seq1, seq2, matrix,
            local=True, gap_penalty=GAP_PENALTY, max_number=1
        )[0].score

    e_values = [
        10 ** estimator.log_evalue(score, SEQ_LENGTH, SEQ_LENGTH * N_SAMPLES)
        for score in TEST_SCORES
    ]
    counts = [
        np.count_nonzero(sample_scores >= score) for score in TEST_SCORES
    ]
    assert e_values == pytest.approx(counts, rel=0.5)


def test_score_scaling(sequences):
    """
    Scaling the substitution scores and gap penalties by a constant
    factor should not influence the obtained E-values.
    Test this by aligning real sequences with a standard and scaled
    scoring scheme and comparing the calculated E-values of these
    alignments.
    """
    SCALING_FACTOR = 1000
    GAP_PENALTY = (-12, -1)
    SEQ_LENGTH = 300

    matrix = align.SubstitutionMatrix.std_protein_matrix()

    np.random.seed(0)
    std_estimator = align.EValueEstimator.from_samples(
        seq.ProteinSequence.alphabet, matrix, GAP_PENALTY,
        BACKGROUND
    )
    scores = [
        align.align_optimal(
            sequences[i], sequences[i+1], matrix, GAP_PENALTY, local=True,
            max_number=1
        )[0].score for i in range(9)
    ]
    std_log_evalues = std_estimator.log_evalue(
        scores, SEQ_LENGTH, SEQ_LENGTH
    )

    scaled_matrix = align.SubstitutionMatrix(
        seq.ProteinSequence.alphabet,
        seq.ProteinSequence.alphabet,
        matrix.score_matrix() * SCALING_FACTOR
    )
    scaled_gap_penalty = (
        GAP_PENALTY[0] * SCALING_FACTOR,
        GAP_PENALTY[1] * SCALING_FACTOR
    )
    scaled_estimator = align.EValueEstimator.from_samples(
        seq.ProteinSequence.alphabet, scaled_matrix, scaled_gap_penalty,
        BACKGROUND
    )
    scores = [
        align.align_optimal(
            sequences[i], sequences[i+1], scaled_matrix, scaled_gap_penalty,
            local=True, max_number=1
        )[0].score for i in range(9)
    ]
    scaled_log_evalues = scaled_estimator.log_evalue(
        scores, SEQ_LENGTH, SEQ_LENGTH
    )

    # Due to relatively low sample size, expect rather large deviation
    assert std_log_evalues.tolist() \
        == pytest.approx(scaled_log_evalues.tolist(), rel=0.2)


def test_invalid_scoring_scheme():
    """
    Check if `from_samples()` raises an exception when the expected
    similarity score between to random symbols is positive.
    """
    alph = seq.ProteinSequence.alphabet
    matrix = align.SubstitutionMatrix(
        alph, alph, np.ones((len(alph), len(alph)), dtype=int)
    )
    # Uniform background frequencies
    freq = np.ones(len(alph))
    
    with pytest.raises(ValueError):
        estimator = EValueEstimator.from_samples(alph, matrix, -10, freq)