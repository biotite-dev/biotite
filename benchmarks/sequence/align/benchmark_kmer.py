import numpy as np
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align

SEQ_LENGTH = 10_000
K = 3


@pytest.fixture(scope="module")
def sequence():
    np.random.seed(0)
    sequence = seq.ProteinSequence()
    sequence.code = np.random.randint(
        len(seq.ProteinSequence.alphabet), size=SEQ_LENGTH
    )
    return sequence


@pytest.fixture(scope="module")
def kmer_alphabet():
    return align.KmerAlphabet(seq.ProteinSequence.alphabet, K)


@pytest.fixture(scope="module")
def matrix():
    return align.SubstitutionMatrix.std_protein_matrix()


@pytest.fixture(scope="module")
def score_threshold_rule(matrix):
    return align.ScoreThresholdRule(matrix, 10)


@pytest.mark.benchmark
def benchmark_create_kmers(kmer_alphabet, sequence):
    """
    Create k-mer codes from a sequence.
    """
    kmer_alphabet.create_kmers(sequence.code)


@pytest.mark.benchmark
def benchmark_similar_kmers(score_threshold_rule, kmer_alphabet):
    """
    Find all k-mers similar to a reference k-mer using a score threshold.
    """
    KMER_CODE = 0

    score_threshold_rule.similar_kmers(kmer_alphabet, KMER_CODE)
