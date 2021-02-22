# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import pickle
import numpy as np
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align


@pytest.fixture
def k():
    return 3

@pytest.fixture
def alphabet():
    return seq.NucleotideSequence.alphabet_unamb

@pytest.fixture
def random_sequences(k, alphabet):
    N_SEQS = 10
    SEQ_LENGTH = 1000

    np.random.seed(0)
    sequences = []
    for _ in range(N_SEQS):
        sequence = seq.NucleotideSequence()
        sequence.code = np.random.randint(len(alphabet), size=SEQ_LENGTH)
        sequences.append(sequence)
    return sequences



def test_add(alphabet, k, random_sequences):
    table = align.KmerTable(alphabet, k)
    for i, sequence in enumerate(random_sequences):
        table.add(sequence, i)
    kmer_alph = table.kmer_alphabet

    for i, sequence in enumerate(random_sequences):
        for j in range(len(sequence) - k + 1):
            kmer = kmer_alph.fuse(sequence.code[j : j+k])
            assert np.array([i,j]) in table[kmer]


def test_merge(alphabet, k, random_sequences):
    """
    Test :meth:`merge()` method by comparing the merge of single
    tables with adding sequences directly into a single table.
    """
    
    ref_table = align.KmerTable(alphabet, k)
    for i, sequence in enumerate(random_sequences):
        ref_table.add(sequence, i)

    test_table = align.KmerTable(alphabet, k)
    for i, sequence in enumerate(random_sequences):
        table = align.KmerTable(alphabet, k)
        table.add(sequence, i)
        test_table.merge(table)
    
    assert test_table == ref_table


def test_get_set(alphabet, k, random_sequences):
    ref_table = align.KmerTable(alphabet, k)
    for i, sequence in enumerate(random_sequences):
        ref_table.add(sequence, i)

    test_table = align.KmerTable(alphabet, k)
    for i in range(len(ref_table)):
        # Copy the positions for each k-mer
        # By calling __getitem__() and __setitem__() 
        test_table[i] = ref_table[i]
    
    assert test_table == ref_table


def test_get_kmers():
    np.random.seed(0)

    alphabet = seq.NucleotideSequence.alphabet_unamb
    table = align.KmerTable(alphabet, 3)

    ref_mask = np.random.choice([False, True], size=len(table))
    ref_kmers = np.where(ref_mask)[0]
    for kmer in ref_kmers:
        # [[0, 0]] = Dummy position
        # The actual position is not relevant for the tested function
        table[kmer] = [[0, 0]]

    test_kmers = table.get_kmers()

    assert test_kmers.tolist() == ref_kmers.tolist()
    

def test_copy(alphabet, k, random_sequences):
    ref_table = align.KmerTable(alphabet, k)
    for i, sequence in enumerate(random_sequences):
        ref_table.add(sequence, i)
    
    test_table = ref_table.copy()

    assert test_table == ref_table


def test_pickle(alphabet, k, random_sequences):
    ref_table = align.KmerTable(alphabet, k)
    for i, sequence in enumerate(random_sequences):
        ref_table.add(sequence, i)
    
    test_table = pickle.loads(pickle.dumps(ref_table))

    assert test_table == ref_table