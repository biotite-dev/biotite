# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import numpy as np
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align


K = 3


@pytest.fixture
def kmer_alphabet():
    return align.KmerAlphabet(seq.ProteinSequence.alphabet, K)



np.random.seed(0)
N = 10
L = 30
@pytest.mark.parametrize(
    "ref_split_kmer_code",
    # Test for single instances as input
    list(np.random.randint(len(seq.ProteinSequence.alphabet), size=(N, K))) +
    # Test for multiple instances as input
    list(np.random.randint(len(seq.ProteinSequence.alphabet), size=(N, L, K)))
)
def test_fuse_and_split(kmer_alphabet, ref_split_kmer_code):
    """
    Check if :meth:`fuse()` and its reverse counterpart :meth:`split()`
    work properly by using them back and forth on random input.
    """
    fused = kmer_alphabet.fuse(ref_split_kmer_code)
    test_split_kmer_code = kmer_alphabet.split(fused)
    
    assert test_split_kmer_code.tolist() == ref_split_kmer_code.tolist()


np.random.seed(0)
N = 10
@pytest.mark.parametrize(
    "split_kmer_code",
    np.random.randint(len(seq.ProteinSequence.alphabet), size=(N, K))
)
def test_encode_and_decode(kmer_alphabet, split_kmer_code):
    """
    Check if :meth:`encode()` and its reverse counterpart
    :meth:`decode()` work properly by using them back and forth on
    random input.
    """
    alph = seq.ProteinSequence.alphabet
    ref_kmer_symbol = alph.decode_multiple(split_kmer_code)
    kmer_code = kmer_alphabet.encode(ref_kmer_symbol)
    test_kmer_symbol = kmer_alphabet.decode(kmer_code)
    
    assert test_kmer_symbol.tolist() == ref_kmer_symbol.tolist()


def test_create_kmers(kmer_alphabet):
    """
    Test :meth:`create_kmers()` against repetitive use of
    :meth:`fuse()`, which rely on two different implementations.
    The input sequence code is randomly created.
    """
    np.random.seed(0)
    LENGTH = 100

    seq_code = np.random.randint(
        len(seq.ProteinSequence.alphabet), size=LENGTH, dtype=np.uint8
    )

    ref_kmers = [
        kmer_alphabet.fuse(seq_code[i : i + kmer_alphabet.k])
        for i in range(len(seq_code) - kmer_alphabet.k + 1)
    ]

    test_kmers = kmer_alphabet.create_kmers(seq_code)

    assert test_kmers.tolist() == ref_kmers