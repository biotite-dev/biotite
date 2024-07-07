# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
import numpy as np
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align


@pytest.mark.parametrize(
    "seed, window, from_sequence, use_permutation",
    itertools.product(range(20), [2, 5, 10, 25], [False, True], [False, True]),
)
def test_minimizer(seed, window, from_sequence, use_permutation):
    """
    Compare the fast minimizer identification algorithm against
    a trivial implementation on randomized input.
    """
    K = 10
    LENGTH = 1000

    sequence = seq.NucleotideSequence(ambiguous=False)
    np.random.seed(seed)
    sequence.code = np.random.randint(len(sequence.alphabet), size=LENGTH)
    kmer_alph = align.KmerAlphabet(sequence.alphabet, K)
    kmers = kmer_alph.create_kmers(sequence.code)

    if use_permutation:
        permutation = align.RandomPermutation()
        order = permutation.permute(kmers)
    else:
        permutation = None
        order = kmers

    # Use an inefficient but simple algorithm for comparison
    ref_minimizer_pos = np.array(
        [np.argmin(order[i : i + window]) + i for i in range(len(order) - (window - 1))]
    )
    # Remove duplicates
    ref_minimizer_pos = np.unique(ref_minimizer_pos)
    ref_minimizers = kmers[ref_minimizer_pos]

    minimizer_selector = align.MinimizerSelector(kmer_alph, window, permutation)
    if from_sequence:
        test_minimizer_pos, test_minimizers = minimizer_selector.select(sequence)
    else:
        test_minimizer_pos, test_minimizers = minimizer_selector.select_from_kmers(
            kmers
        )

    assert test_minimizer_pos.tolist() == ref_minimizer_pos.tolist()
    assert test_minimizers.tolist() == ref_minimizers.tolist()


@pytest.mark.parametrize(
    "seed, s, offset, from_sequence, use_permutation",
    itertools.product(
        range(20),
        [2, 3, 5, 7],
        [(0,), (0, 1, 2), (0, -1), (-2, -1)],
        [False, True],
        [False, True],
    ),
    # Print tuples in name of test
    ids=lambda x: str(x).replace(" ", "") if isinstance(x, tuple) else None,
)
def test_syncmer(seed, s, offset, from_sequence, use_permutation):
    """
    Compare the fast syncmer identification algorithm against
    a trivial implementation on randomized input.
    """
    K = 10
    LENGTH = 1000

    sequence = seq.NucleotideSequence(ambiguous=False)
    alphabet = sequence.alphabet
    np.random.seed(seed)
    sequence.code = np.random.randint(len(alphabet), size=LENGTH)

    kmers = align.KmerAlphabet(alphabet, K).create_kmers(sequence.code)
    smers = align.KmerAlphabet(alphabet, s).create_kmers(sequence.code)

    if use_permutation:
        permutation = align.RandomPermutation()
        order = permutation.permute(smers)
    else:
        permutation = None
        order = smers

    # Use an inefficient but simple algorithm for comparison
    ref_syncmer_pos = []
    for i in range(len(kmers)):
        window = K - s + 1
        order_in_kmer = order[i : i + window]
        min_smer_pos = np.argmin(order_in_kmer)
        # Wraparound negative indices -> modulo operation
        if np.isin(min_smer_pos, np.array(offset) % window):
            ref_syncmer_pos.append(i)
    ref_syncmer_pos = np.array(ref_syncmer_pos, dtype=np.int64)
    ref_syncmers = kmers[ref_syncmer_pos]

    syncmer_selector = align.SyncmerSelector(
        sequence.alphabet, K, s, permutation, offset
    )
    if from_sequence:
        test_syncmer_pos, test_syncmers = syncmer_selector.select(sequence)
    else:
        test_syncmer_pos, test_syncmers = syncmer_selector.select_from_kmers(kmers)

    assert test_syncmer_pos.tolist() == ref_syncmer_pos.tolist()
    assert test_syncmers.tolist() == ref_syncmers.tolist()


def test_cached_syncmer():
    """
    Check if :class:`CachedSyncmerSelector` gives the same results as
    :class:`SyncmerSelector` for randomized input.

    This is not included :func:`test_syncmer()` as
    :class:`CachedSyncmerSelector` creation takes quite long and hence
    would bloat the test run time due to the large parametrization
    matrix of :func:`test_syncmer()`.
    """
    K = 5
    S = 2
    LENGTH = 1000

    sequence = seq.NucleotideSequence(ambiguous=False)
    np.random.seed(0)
    sequence.code = np.random.randint(len(sequence.alphabet), size=LENGTH)

    syncmer_selector = align.SyncmerSelector(sequence.alphabet, K, S)
    ref_syncmer_pos, ref_syncmers = syncmer_selector.select(sequence)

    cached_syncmer_selector = align.CachedSyncmerSelector(sequence.alphabet, K, S)
    test_syncmer_pos, test_syncmers = cached_syncmer_selector.select(sequence)

    assert test_syncmer_pos.tolist() == ref_syncmer_pos.tolist()
    assert test_syncmers.tolist() == ref_syncmers.tolist()


@pytest.mark.parametrize(
    "offset, exception_type",
    [
        # Duplicate values
        ((1, 1), ValueError),
        ((0, 2, 0), ValueError),
        ((0, -10), ValueError),
        # Offset out of window range
        ((-11,), IndexError),
        ((10,), IndexError),
    ],
)
def test_syncmer_invalid_offset(offset, exception_type):
    """
    Test whether known invalid offsets lead to the expected exception.
    """
    K = 11
    S = 2
    with pytest.raises(exception_type):
        align.SyncmerSelector(
            # Any alphabet would work here
            seq.NucleotideSequence.alphabet_unamb,
            K,
            S,
            offset=offset,
        )


@pytest.mark.parametrize("use_permutation", [False, True])
def test_mincode(use_permutation):
    """
    Simple test whether :class:`MincodeSelector` selects *k-mers* below
    the threshold value and whether the compression factor is correctly
    used.
    """
    K = 5
    COMPRESSION = 4

    kmer_alph = align.KmerAlphabet(seq.NucleotideSequence.alphabet_unamb, K)
    np.random.seed(0)
    kmers = np.arange(len(kmer_alph), dtype=np.int64)

    if use_permutation:
        permutation = align.RandomPermutation()
        permutation_offset = permutation.min
        permutation_range = permutation.max - permutation.min + 1
        order = permutation.permute(kmers)
    else:
        permutation = None
        permutation_offset = 0
        permutation_range = len(kmer_alph)
        order = kmers

    mincode_selector = align.MincodeSelector(kmer_alph, COMPRESSION, permutation)

    _, mincode_pos = mincode_selector.select_from_kmers(kmers)
    threshold = permutation_offset + permutation_range / COMPRESSION
    assert mincode_pos.tolist() == np.where(order < threshold)[0].tolist()
    assert len(mincode_pos) * COMPRESSION == pytest.approx(len(kmers), rel=0.02)
