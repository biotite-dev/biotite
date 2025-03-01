# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import functools
import itertools
import pickle
import string
import numpy as np
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align


class FixedBucketKmerTable:
    """
    A wrapper around :class:`BucketKmerTable` with a fixed number of
    buckets.
    This allows test functions to call static functions from
    :class:`KmerTable` and :class:`FixedBinnedKmerTable` with the same
    signature, avoiding if-else constructs.
    """

    def __init__(self, n_buckets):
        self._n_buckets = n_buckets

    def __getattr__(self, name):
        attr = getattr(align.BucketKmerTable, name)
        if attr.__name__ in ["from_sequences", "from_kmers", "from_kmer_selection"]:
            return functools.partial(attr, n_buckets=self._n_buckets)
        else:
            return attr

    def __repr__(self):
        return f"BucketKmerTable({self._n_buckets})"


def idfn(val):
    if isinstance(val, FixedBucketKmerTable):
        return repr(val)


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


@pytest.mark.parametrize(
    "spacing, table_class",
    itertools.product(
        [None, "10111011011", "1001111010101"],
        [
            align.KmerTable,
            # Choose number of buckets, so that there is one test case
            # with less buckets than number of possible kmers ...
            FixedBucketKmerTable(1000),
            # ... and one test case with more buckets (perfect hashing)
            FixedBucketKmerTable(1000000),
        ],
    ),
    ids=idfn,
)
def test_from_sequences(k, random_sequences, spacing, table_class):
    """
    Test the :meth:`from_sequences()` constructor, by checking for each
    sequence position, if the position is in the C-array of the
    corresponding k-mer.
    """
    table = table_class.from_sequences(k, random_sequences, spacing=spacing)
    kmer_alph = align.KmerAlphabet(random_sequences[0].alphabet, k, spacing)
    assert kmer_alph == table.kmer_alphabet

    for i, sequence in enumerate(random_sequences):
        for j in range(kmer_alph.kmer_array_length(len(sequence))):
            if spacing is None:
                kmer = kmer_alph.fuse(sequence.code[j : j + k])
            else:
                kmer = kmer_alph.fuse(sequence.code[kmer_alph.spacing + j])
            assert np.array([i, j]) in table[kmer]


@pytest.mark.parametrize(
    "table_class",
    [align.KmerTable, FixedBucketKmerTable(1000), FixedBucketKmerTable(1000000)],
    ids=idfn,
)
def test_from_kmers(k, random_sequences, table_class):
    """
    Test the :meth:`from_kmers()` constructor, by comparing it with an
    equivalent table created with :meth:`from_sequences()`.
    """
    ref_table = table_class.from_sequences(k, random_sequences)
    kmer_alph = ref_table.kmer_alphabet

    kmer_arrays = [
        kmer_alph.create_kmers(sequence.code) for sequence in random_sequences
    ]
    test_table = table_class.from_kmers(kmer_alph, kmer_arrays)

    assert test_table == ref_table


@pytest.mark.parametrize(
    "table_class",
    [align.KmerTable, FixedBucketKmerTable(1000), FixedBucketKmerTable(1000000)],
    ids=idfn,
)
def test_from_kmer_selection(k, alphabet, random_sequences, table_class):
    """
    Test the :meth:`test_from_kmer_selection()` constructor, by checking
    for each stored k-mer position, whether it is found at that position
    in the sequences.
    """
    N_POSITIONS = 100

    kmer_alph = align.KmerAlphabet(alphabet, k)
    kmer_arrays = [
        kmer_alph.create_kmers(sequence.code) for sequence in random_sequences
    ]
    np.random.seed(0)
    filtered_pos_arrays = [
        np.random.randint(len(kmers), size=N_POSITIONS) for kmers in kmer_arrays
    ]
    filtered_kmer_arrays = [
        kmers[filtered_pos]
        for filtered_pos, kmers in zip(filtered_pos_arrays, kmer_arrays)
    ]
    kmer_table = table_class.from_kmer_selection(
        kmer_alph, filtered_pos_arrays, filtered_kmer_arrays
    )

    # The total number of k-mers in the table
    # should be the total number of input k-mers
    assert np.sum(kmer_table.count(np.arange(len(kmer_alph)))) == np.sum(
        [len(kmers) for kmers in filtered_kmer_arrays]
    )
    # Each k-mer in the table should be found
    # in the original k-mer sequences
    for kmer in range(len(kmer_alph)):
        for ref_id, pos in kmer_table[kmer]:
            assert kmer_arrays[ref_id][pos] == kmer


@pytest.mark.parametrize(
    "table_class",
    [align.KmerTable, FixedBucketKmerTable(1000), FixedBucketKmerTable(1000000)],
    ids=idfn,
)
def test_from_tables(k, random_sequences, table_class):
    """
    Test :meth:`from_tables()` constructor by comparing the merge of
    single tables with adding sequences directly into a single table.
    """
    ref_table = table_class.from_sequences(
        k, random_sequences, np.arange(len(random_sequences))
    )

    tables = [
        table_class.from_sequences(k, [sequence], [i])
        for i, sequence in enumerate(random_sequences)
    ]
    test_table = table_class.from_tables(tables)

    assert test_table == ref_table


def test_from_positions(k, random_sequences):
    """
    Test :meth:`from_positions()` constructor by comparing it with an
    equivalent table created with :meth:`from_sequences()`.
    """
    ref_table = align.KmerTable.from_sequences(k, random_sequences)

    kmer_dict = {kmer: ref_table[kmer] for kmer in range(len(ref_table))}
    test_table = align.KmerTable.from_positions(ref_table.kmer_alphabet, kmer_dict)

    assert test_table == ref_table


@pytest.mark.parametrize(
    "table_class, use_similarity_rule",
    itertools.product(
        [align.KmerTable, FixedBucketKmerTable(1000), FixedBucketKmerTable(10000000)],
        [False, True],
    ),
    ids=idfn,
)
def test_match_table(table_class, use_similarity_rule):
    """
    Test the :meth:`match_table()` method based on a known example.

    Using the similarity rule should give the same result, as it is
    chosen to yield only the same k-mer as similar k-mer.
    """
    alphabet = seq.LetterAlphabet(string.ascii_lowercase + "_")
    phrase1 = "how_much_wood_would_a_woodchuck_chuck_if_a_woodchuck_could_chuck_wood"
    phrase2 = "woodchuck"
    sequence1 = seq.GeneralSequence(alphabet, phrase1)
    sequence2 = seq.GeneralSequence(alphabet, phrase2)

    rule = _identity_rule(alphabet) if use_similarity_rule else None

    table1 = table_class.from_sequences(4, [sequence1])
    table2 = table_class.from_sequences(4, [sequence2])

    ref_matches = set(
        [
            (0, 9),
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
        ]
    )

    test_matches = table1.match_table(table2, similarity_rule=rule)
    # the reference indices are irrelevant for this test
    test_matches = test_matches[:, [1, 3]]
    test_matches = set([tuple(match) for match in test_matches])
    assert test_matches == ref_matches


@pytest.mark.parametrize(
    "table_class, use_similarity_rule",
    itertools.product(
        [align.KmerTable, FixedBucketKmerTable(1000), FixedBucketKmerTable(1000000)],
        [False, True],
    ),
    ids=idfn,
)
def test_match(k, random_sequences, table_class, use_similarity_rule):
    """
    Test the :meth:`match()` method compared to a manual,
    inefficient approach using indexing.

    Using the similarity rule should give the same result, as it is
    chosen to yield only the same k-mer as similar k-mer.
    """
    query_sequence = random_sequences[0]
    table_sequences = random_sequences[1:]
    table = table_class.from_sequences(k, table_sequences)

    kmers = table.kmer_alphabet.create_kmers(query_sequence.code)
    ref_matches = []
    for i, kmer in enumerate(kmers):
        matches = table[kmer]
        matches = np.stack(
            [np.full(len(matches), i, dtype=np.uint32), matches[:, 0], matches[:, 1]],
            axis=1,
        )
        ref_matches.append(matches)
    ref_matches = np.concatenate(ref_matches)

    rule = _identity_rule(table.alphabet) if use_similarity_rule else None
    test_matches = table.match(query_sequence, similarity_rule=rule)

    assert np.array_equal(test_matches.tolist(), ref_matches.tolist())


@pytest.mark.parametrize(
    "table_class",
    [align.KmerTable, FixedBucketKmerTable(1000), FixedBucketKmerTable(1000000)],
    ids=idfn,
)
def test_match_kmer_selection(k, random_sequences, table_class):
    """
    Same as :func:`test_match()` but with a subset of positions.
    """
    N_POS = 100

    query_sequence = random_sequences[0]
    table_sequences = random_sequences[1:]
    table = table_class.from_sequences(k, table_sequences)

    kmers = table.kmer_alphabet.create_kmers(query_sequence.code)
    np.random.seed(0)
    positions = np.random.randint(len(kmers), size=N_POS)
    ref_matches = []
    for pos in positions:
        kmer = kmers[pos]
        matches = table[kmer]
        matches = np.stack(
            [np.full(len(matches), pos, dtype=np.uint32), matches[:, 0], matches[:, 1]],
            axis=1,
        )
        ref_matches.append(matches)
    ref_matches = np.concatenate(ref_matches)

    test_matches = table.match_kmer_selection(positions, kmers[positions])

    assert np.array_equal(test_matches.tolist(), ref_matches.tolist())


@pytest.mark.parametrize(
    "table_class, use_mask",
    itertools.product(
        [align.KmerTable, FixedBucketKmerTable(1000), FixedBucketKmerTable(1000000)],
        [False, True],
    ),
    ids=idfn,
)
def test_match_equivalence(k, random_sequences, table_class, use_mask):
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

    table = table_class.from_sequences(k, table_sequences, ignore_masks=table_masks)

    # 42 -> Dummy value that is distinct from all reference indices
    ref_table = table_class.from_sequences(
        k, [query_sequence], [42], ignore_masks=[query_mask]
    )
    ref_matches = table.match_table(ref_table)
    assert np.all(ref_matches[:, 0] == 42)
    # Store matches in set to remove the order dependency
    # The first column is not present in the matches
    # returned by 'match_sequence()' -> [:, 1:]
    ref_matches = set([tuple(match) for match in ref_matches[:, 1:]])

    test_matches = table.match(query_sequence, ignore_mask=query_mask)
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
    ],
    ids = idfn
)  # fmt: skip
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
    table = align.KmerTable.from_sequences(k, [sequence], ignore_masks=[input_mask])

    # Get the k-mer positions that were masked
    test_output_mask = np.zeros(len(ref_output_mask), dtype=bool)
    for kmer in table.get_kmers():
        seq_indices = table[kmer][:, 1]
        test_output_mask[seq_indices] = True

    assert test_output_mask.tolist() == ref_output_mask.tolist()


@pytest.mark.parametrize(
    "table_class, selected_kmers",
    [
        (align.KmerTable, False),
        (align.KmerTable, True),
        (FixedBucketKmerTable(1000), True),
        (FixedBucketKmerTable(1000000), True),
    ],
    ids=idfn,
)
def test_count(k, random_sequences, table_class, selected_kmers):
    """
    Test :meth:`count()` against an inefficient solution using a list
    comprehension.
    """
    N_KMERS = 100

    table = table_class.from_sequences(k, random_sequences)

    if selected_kmers:
        np.random.seed(0)
        kmers = np.random.randint(len(table.kmer_alphabet), size=N_KMERS)
        ref_counts = [len(table[kmer]) for kmer in kmers]
        test_counts = table.count(kmers)
    else:
        ref_counts = [len(table[kmer]) for kmer in range(len(table.kmer_alphabet))]
        test_counts = table.count()

    assert test_counts.tolist() == ref_counts


@pytest.mark.parametrize(
    "table_class",
    [align.KmerTable, FixedBucketKmerTable(1000), FixedBucketKmerTable(1000000)],
    ids=idfn,
)
def test_get_kmers(table_class):
    """
    Test whether the correct used *k-mers* are returned by
    :meth:`get_kmers()`, by constructing a table with exactly one
    appearance for each *k-mer* in a random list of *k-mers*.
    """
    np.random.seed(0)

    kmer_alphabet = align.KmerAlphabet(seq.NucleotideSequence.unambiguous_alphabet(), 8)
    ref_mask = np.random.choice([False, True], size=len(kmer_alphabet))
    ref_kmers = np.where(ref_mask)[0]
    table = table_class.from_kmers(kmer_alphabet, [ref_kmers])

    test_kmers = table.get_kmers()

    assert test_kmers.tolist() == ref_kmers.tolist()


@pytest.mark.parametrize(
    "table_class",
    [align.KmerTable, FixedBucketKmerTable(1000), FixedBucketKmerTable(1000000)],
    ids=idfn,
)
def test_pickle(k, random_sequences, table_class):
    """
    Test support of the pickle protocol by pickling and unpickling
    a :class:`KmerTable` and checking if the object is still equal to
    the original one.
    """
    ref_table = table_class.from_sequences(k, random_sequences)

    test_table = pickle.loads(pickle.dumps(ref_table))

    assert test_table == ref_table


@pytest.mark.parametrize(
    "n_kmers, load_factor",
    itertools.product([1_000, 100_000, 10_000_000, 1_000_000_000], [0.2, 1.0, 2.0]),
)
def test_bucket_number(n_kmers, load_factor):
    """
    Check if the number of buckets is always maximum five percent larger
    than the load factor corrected number of *k-mers*.
    Use larger bucket numbers for this to ensure there even exists a
    prime within the deviation threshold
    """
    min_n_buckets = int(n_kmers / load_factor)
    test_n_buckets = align.bucket_number(n_kmers, load_factor)

    assert test_n_buckets >= min_n_buckets
    assert test_n_buckets <= min_n_buckets * 1.05


def _identity_rule(alphabet):
    score_matrix = np.full((len(alphabet),) * 2, -1, dtype=int)
    np.fill_diagonal(score_matrix, 0)
    matrix = align.SubstitutionMatrix(alphabet, alphabet, score_matrix)
    rule = align.ScoreThresholdRule(matrix, 0)
    return rule
