# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import string
import pickle
import numpy as np
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align


@pytest.fixture
def k():
    return 8

@pytest.fixture
def alphabet():
    return seq.NucleotideSequence.unambiguous_alphabet()

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



@pytest.mark.parametrize("spacing", [None, "10111011011", "1001111010101"])
def test_from_sequences(k, random_sequences, spacing):
    """
    Test the :meth:`from_sequences()` constructor, by checking for each
    sequence position, if the position is in the C-array of the
    corresponding k-mer.
    """
    table = align.KmerTable.from_sequences(
        k, random_sequences, spacing=spacing
    )
    kmer_alph = align.KmerAlphabet(random_sequences[0].alphabet, k, spacing)
    assert kmer_alph == table.kmer_alphabet

    for i, sequence in enumerate(random_sequences):
        for j in range(kmer_alph.kmer_array_length(len(sequence))):
            if spacing is None:
                kmer = kmer_alph.fuse(sequence.code[j : j+k])
            else:
                kmer = kmer_alph.fuse(sequence.code[kmer_alph.spacing + j])
            assert np.array([i,j]) in table[kmer]



def test_from_kmers(k, random_sequences):
    """
    Test the :meth:`from_kmers()` constructor, by comparing it with an
    equivalent table created with :meth:`from_sequences()`.
    """
    ref_table = align.KmerTable.from_sequences(k, random_sequences)
    kmer_alph = ref_table.kmer_alphabet

    kmer_arrays = [
        kmer_alph.create_kmers(sequence.code) for sequence in random_sequences
    ]
    test_table = align.KmerTable.from_kmers(kmer_alph, kmer_arrays)
    
    assert test_table == ref_table


def test_from_tables(k, random_sequences):
    """
    Test :meth:`from_tables()` constructor by comparing the merge of
    single tables with adding sequences directly into a single table.
    """
    
    ref_table = align.KmerTable.from_sequences(
        k, random_sequences, np.arange(len(random_sequences))
    )

    tables = [
        align.KmerTable.from_sequences(k, [sequence], [i])
        for i, sequence in enumerate(random_sequences)
    ]
    test_table = align.KmerTable.from_tables(tables)
    
    assert test_table == ref_table


def test_from_positions(k, random_sequences):
    """
    Test :meth:`from_positions()` constructor by comparing it with an
    equivalent table created with :meth:`from_sequences()`.
    """
    ref_table = align.KmerTable.from_sequences(k, random_sequences)

    kmer_dict = {kmer : ref_table[kmer] for kmer in range(len(ref_table))}
    test_table = align.KmerTable.from_positions(
        ref_table.kmer_alphabet, kmer_dict
    )
    
    assert test_table == ref_table


@pytest.mark.parametrize("use_similarity_rule", [False, True])
def test_match_table(use_similarity_rule):
    """
    Test the :meth:`match_table()` method based on a known example.
    
    Using the similarity rule should give the same result, as it is
    chosen to yield only the same k-mer as similar k-mer.
    """
    alphabet = seq.LetterAlphabet(string.ascii_lowercase + "_")
    phrase1 = "how_much_wood_would_a_woodchuck_chuck_if_a_woodchuck_could_" \
              "chuck_wood"
    phrase2 = "woodchuck"
    sequence1 = seq.GeneralSequence(alphabet, phrase1)
    sequence2 = seq.GeneralSequence(alphabet, phrase2)

    rule = _identity_rule(alphabet) if use_similarity_rule else None
    
    table1 = align.KmerTable.from_sequences(4, [sequence1])
    table2 = align.KmerTable.from_sequences(4, [sequence2])

    ref_matches = set([
        (0,  9),
        (0, 22),
        (1, 23),
        (2, 24),
        (3, 25),
        (4, 26),
        (5, 27),
        (4, 32),
        (5, 33),
        (0, 43),
        (1, 44),
        (2, 45),
        (3, 46),
        (4, 47),
        (5, 48),
        (4, 59),
        (5, 60),
        (0, 65),
    ])

    test_matches = table1.match_table(table2, similarity_rule=rule)
    # the reference indices are irrelevant for this test
    test_matches = test_matches[:, [1,3]]
    test_matches = set([tuple(match) for match in test_matches])
    assert test_matches == ref_matches


@pytest.mark.parametrize("use_similarity_rule", [False, True])
def test_match(k, random_sequences, use_similarity_rule):
    """
    Test the :meth:`match()` method compared to a manual,
    inefficient approach using indexing.

    Using the similarity rule should give the same result, as it is
    chosen to yield only the same k-mer as similar k-mer.
    """
    query_sequence = random_sequences[0]
    table_sequences = random_sequences[1:]   
    table = align.KmerTable.from_sequences(k, table_sequences)
    
    kmers = table.kmer_alphabet.create_kmers(query_sequence.code)
    ref_matches = []
    for i, kmer in enumerate(kmers):
        matches = table[kmer]
        matches = np.stack(
            [
                np.full(len(matches), i, dtype=np.uint32), 
                matches[:,0],
                matches[:,1]
            ],
            axis=1
        )
        ref_matches.append(matches)
    ref_matches = np.concatenate(ref_matches)

    rule = _identity_rule(table.alphabet) if use_similarity_rule else None
    test_matches = table.match(query_sequence, similarity_rule=rule)

    assert np.array_equal(test_matches.tolist(), ref_matches.tolist())


@pytest.mark.parametrize("use_mask", [False, True])
def test_match_equivalence(k, random_sequences, use_mask):
    """
    Check if both, the :meth:`match()` and meth:`match_table()`
    method, find the same matches.
    """
    query_sequence = random_sequences[0]
    table_sequences = random_sequences[1:]

    if use_mask:
        # Create a random removal mask with low removal probability
        np.random.seed(0)
        removal_masks = [
            np.random.choice([False, True], size=len(sequence), p=[0.95, 0.05])
            for sequence in random_sequences
        ]
    else:
        removal_masks = [None] * len(random_sequences)
    query_mask = removal_masks[0]
    table_masks = removal_masks[1:]
    
    table = align.KmerTable.from_sequences(
        k, table_sequences, ignore_masks=table_masks
    )
    
    # 42 -> Dummy value that is distinct from all reference indices
    ref_table = align.KmerTable.from_sequences(
        k, [query_sequence], [42], ignore_masks=[query_mask]
    )
    ref_matches = table.match_table(ref_table)
    assert np.all(ref_matches[:,0] == 42)
    # Store matches in set to remove the order dependency
    # The first column is not present in the matches
    # returned by 'match_sequence()' -> [:, 1:]
    ref_matches = set([tuple(match) for match in ref_matches[:, 1:]])

    test_matches = table.match(
        query_sequence, ignore_mask=query_mask
    )
    test_matches = set([tuple(match) for match in test_matches])


    # Check if any match is found at all
    assert len(ref_matches) > 0
    # The first column is not present in 'test_matches'
    assert test_matches == ref_matches


@pytest.mark.parametrize(
    "k, input_mask, ref_output_mask",
    [
        (
            3,
            [0,1,1,0,0,0,0,0,1],
            [0,0,0,1,1,1,0]
        ),
        (
            4,
            [1,0,0,0,0,1,0,0,0,0,0,1,0,0],
            [0,1,0,0,0,0,1,1,0,0,0]
        ),
    ]
)
def test_masking(k, input_mask, ref_output_mask):
    """
    Explicitly test the conversion of removal masks to k-mer masks
    using known examples.
    Since the conversion function is private, this is tested indirectly,
    by looking at the sequence positions, that were added to the array.
    """
    input_mask = np.array(input_mask, dtype=bool)
    ref_output_mask = np.array(ref_output_mask, dtype=bool)

    sequence = seq.NucleotideSequence()
    sequence.code = np.zeros(len(input_mask))
    table = align.KmerTable.from_sequences(
        k, [sequence], ignore_masks=[input_mask]
    )

    # Get the k-mer positions that were masked
    test_output_mask = np.zeros(len(ref_output_mask), dtype=bool)
    for kmer in table.get_kmers():
        seq_indices = table[kmer][:, 1]
        test_output_mask[seq_indices] = True
    
    assert test_output_mask.tolist() == ref_output_mask.tolist()


def test_get_kmers():
    """
    Test whether the correct used *k-mers* are returned by
    :meth:`get_kmer()`, by constructing a table with exactly one
    appearance for each *k-mer* in a random list of *k-mers*.
    """
    np.random.seed(0)

    kmer_alphabet = align.KmerAlphabet(
        seq.NucleotideSequence.unambiguous_alphabet(),
        8
    )
    ref_mask = np.random.choice([False, True], size=len(kmer_alphabet))
    ref_kmers = np.where(ref_mask)[0]
    kmer_dict = {kmer: np.zeros((1, 2), dtype=np.uint32) for kmer in ref_kmers}
    # [[0, 0]] = Dummy position
    # The actual position is not relevant for the tested function
    table = align.KmerTable.from_positions(kmer_alphabet, kmer_dict)

    test_kmers = table.get_kmers()

    assert test_kmers.tolist() == ref_kmers.tolist()


def test_pickle(alphabet, k, random_sequences):
    """
    Test support of the pickle protocol by pickling and unpickling
    a :class:`KmerTable` and checking if the object is still equal to
    the original one.
    """
    ref_table = align.KmerTable.from_sequences(k, random_sequences)
    
    test_table = pickle.loads(pickle.dumps(ref_table))

    assert test_table == ref_table


def _identity_rule(alphabet):
    score_matrix = np.full((len(alphabet),) * 2, -1, dtype=int)
    np.fill_diagonal(score_matrix, 0)
    matrix = align.SubstitutionMatrix(alphabet, alphabet, score_matrix)
    rule = align.ScoreThresholdRule(matrix, 0)
    return rule