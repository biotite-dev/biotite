# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import numpy as np
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align


def _create_frequency_permutation(k):
    """
    Convenience function to create FrequencyPermutation from *k* instead
    of a `KmerAlphabet`
    """
    kmer_alph = align.KmerAlphabet(seq.NucleotideSequence.alphabet_unamb, k)
    return align.FrequencyPermutation(kmer_alph, np.arange(len(kmer_alph)))


def test_random_permutation_modulo():
    """
    Test if the LCG of :class:`RandomPermutation`, omitting the explicit
    modulo due to automatic bit truncation, gives the same result as
    a slower implementation using explicit modulo and an unlimited
    integer type.
    """
    LCG_A = align.RandomPermutation.LCG_A
    LCG_C = align.RandomPermutation.LCG_C
    LCG_M = int(2**64)
    SEQ_LENGTH = 10_000

    np.random.seed(0)
    kmers = np.random.randint(
        np.iinfo(np.int64).max + 1, size=SEQ_LENGTH, dtype=np.int64
    )

    ref_order = [(LCG_A * kmer.item() + LCG_C) % LCG_M for kmer in kmers]

    permutation = align.RandomPermutation()
    test_order = permutation.permute(kmers)
    # Convert back to uint64
    # to revert additional step in RandomPermutation
    test_order = test_order.view(np.uint64)

    assert test_order.tolist() == ref_order


def test_random_permutation_randomness():
    """
    A simple test of randomness from :class:`RandomPermutation`:
    The permutation of an arbitrary range of positive values should
    have an almost equal number of positive and negative integers.
    """
    SEQ_LENGTH = 10_000_000
    FRAME_SIZE = 20

    kmers = np.arange(0, SEQ_LENGTH, dtype=np.int64)
    permutation = align.RandomPermutation()
    order = permutation.permute(kmers)
    positive = np.sign(order) == 1
    n_positive = np.convolve(positive, np.ones(FRAME_SIZE), mode="valid")
    distribution, _ = np.histogram(n_positive, bins=np.arange(0, 10 * FRAME_SIZE))

    # Since each value in the k-mer array is unique,
    # all mapped values should be unique as well
    assert len(np.unique(order)) == SEQ_LENGTH
    # Peak of distribution should be at FRAME_SIZE / 2 since an equal
    # number of positive and negative integers are expected
    assert np.argmax(distribution) == FRAME_SIZE // 2


def test_frequency_permutation():
    K = 5
    kmer_alphabet = align.KmerAlphabet(seq.NucleotideSequence.alphabet_unamb, K)
    np.random.seed(0)
    # Generate a random count order for each k-mer
    # Use 'np.arange()' to generate a unique order,
    # but also use step != 1 to make the count != order
    counts = np.arange(2 * len(kmer_alphabet), step=2)
    np.random.shuffle(counts)
    kmer_table = align.KmerTable.from_positions(
        kmer_alphabet,
        # The actual k-mer positions are dummy values,
        # only the number of each k-mer is important for this test
        {i: np.zeros((count, 2)) for i, count in enumerate(counts)},
    )
    permutation = align.FrequencyPermutation.from_table(kmer_table)

    kmers_sorted_by_frequency = np.argsort(counts)
    assert (
        permutation.permute(kmers_sorted_by_frequency).tolist()
        == np.arange(len(kmer_alphabet), dtype=np.int64).tolist()
    )


@pytest.mark.parametrize(
    "kmer_range, permutation",
    [
        (np.iinfo(np.int64).max, align.RandomPermutation()),
        (int(4**5), _create_frequency_permutation(5)),
        (int(4**8), _create_frequency_permutation(8)),
    ],
)
def test_min_max(kmer_range, permutation):
    """
    Test whether the permutation range (min and max) is set correctly,
    by checking that no returned order value is outside that range,
    and at least one of them is very close (controlled by a tolerance
    parameter) to each end of the range.
    """
    TOLERANCE = 0.001
    N_KMERS = 100_000

    np.random.seed(0)
    kmers = np.random.randint(kmer_range, size=N_KMERS, dtype=np.int64)
    order = permutation.permute(kmers)

    assert permutation.max > permutation.min
    # No value should be outside range
    assert np.all(order >= permutation.min)
    assert np.all(order <= permutation.max)
    # At least of value should be near end of range
    permutation_range = permutation.max - permutation.min + 1
    assert np.any(order <= permutation.min + permutation_range * TOLERANCE)
    assert np.any(order >= permutation.max - permutation_range * TOLERANCE)
