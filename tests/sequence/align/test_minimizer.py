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
    itertools.product(
        range(20),
        [2, 5, 10, 25],
        [False, True],
        [False, True]
    )
)
def test_minimize(seed, window, from_sequence, use_permutation):
    """
    Compare the fast minimizer identification algorithm against
    a trivial implementation based on randomized input.
    """
    K = 10
    LENGTH = 1000

    sequence = seq.NucleotideSequence(ambiguous=False)
    np.random.seed(seed)
    sequence.code = np.random.randint(len(sequence.alphabet), size=LENGTH)
    kmer_alph = align.KmerAlphabet(sequence.alphabet, K)
    kmers = kmer_alph.create_kmers(sequence.code)

    if use_permutation:
        permutation = align.RandomPermutation(kmer_alph)
        order = permutation.permute(kmers)
    else:
        permutation = None
        order = kmers

    # Use an inefficient but simple algorithm for comparison
    ref_minimizer_pos = np.array([
        np.argmin(order[i : i + window]) + i
        for i in range(len(order) - (window - 1))
    ])
    # Remove duplicates
    ref_minimizer_pos = np.unique(ref_minimizer_pos)
    ref_minimizers = kmers[ref_minimizer_pos]
    
    minimizer_rule = align.MinimizerRule(kmer_alph, window, permutation)
    if from_sequence:
        test_minimizer_pos, test_minimizers \
            = minimizer_rule.select(sequence)
    else:
        test_minimizer_pos, test_minimizers \
            = minimizer_rule.select_from_kmers(kmers)
    
    assert test_minimizer_pos.tolist() == ref_minimizer_pos.tolist()
    assert test_minimizers.tolist() == ref_minimizers.tolist()
