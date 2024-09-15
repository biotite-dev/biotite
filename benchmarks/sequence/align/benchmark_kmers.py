import functools
import pickle
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


@pytest.fixture(scope="module", params=[None, "11*11*1*1***111"])
def kmer_alphabet(request):
    return align.KmerAlphabet(
        seq.NucleotideSequence.unambiguous_alphabet(), k=9, spacing=request.param
    )


@pytest.fixture(scope="module")
def seq_code():
    LENGTH = 1000

    rng = np.random.default_rng(0)
    return rng.integers(
        len(seq.NucleotideSequence.unambiguous_alphabet()), size=LENGTH, dtype=np.uint8
    )


@pytest.fixture(scope="module")
def kmer_code(kmer_alphabet, seq_code):
    return kmer_alphabet.create_kmers(seq_code)


@pytest.fixture(
    scope="module", params=[align.KmerTable, FixedBucketKmerTable(10000)], ids=idfn
)
def kmer_table(kmer_alphabet, request):
    N_SEQUENCES = 100
    LENGTH = 1000

    Table = request.param

    rng = np.random.default_rng(0)
    seq_codes = [
        rng.integers(
            len(seq.NucleotideSequence.unambiguous_alphabet()),
            size=LENGTH,
            dtype=np.uint8,
        )
        for _ in range(N_SEQUENCES)
    ]

    kmers = [kmer_alphabet.create_kmers(seq_code) for seq_code in seq_codes]
    return Table.from_kmers(kmer_alphabet, kmers)


@pytest.mark.benchmark
def benchmark_kmer_decomposition(kmer_alphabet, seq_code):
    kmer_alphabet.create_kmers(seq_code)


@pytest.mark.parametrize(
    "Table",
    [
        align.KmerTable,
        FixedBucketKmerTable(100000),
    ],
    ids=idfn,
)
@pytest.mark.benchmark
def benchmark_indexing_from_sequences(kmer_alphabet, seq_code, Table):
    N_SEQUENCES = 100

    sequence = seq.NucleotideSequence()
    sequence.code = seq_code
    sequences = [sequence] * N_SEQUENCES
    Table.from_sequences(kmer_alphabet.k, sequences, spacing=kmer_alphabet.spacing)


@pytest.mark.parametrize(
    "Table",
    [
        align.KmerTable,
        FixedBucketKmerTable(100000),
    ],
    ids=idfn,
)
@pytest.mark.benchmark
def benchmark_indexing_from_kmers(kmer_alphabet, seq_code, Table):
    N_SEQUENCES = 100

    kmers = [kmer_alphabet.create_kmers(seq_code)] * N_SEQUENCES
    Table.from_kmers(kmer_alphabet, kmers)


@pytest.mark.parametrize(
    "Table",
    [
        align.KmerTable,
        FixedBucketKmerTable(100000),
    ],
    ids=idfn,
)
@pytest.mark.benchmark
def benchmark_indexing_from_kmer_selection(kmer_alphabet, kmer_code, Table):
    N_SEQUENCES = 100

    kmers = [kmer_code] * N_SEQUENCES
    positions = [np.arange(len(kmer_code), dtype=np.uint32)] * N_SEQUENCES
    Table.from_kmer_selection(kmer_alphabet, positions, kmers)


@pytest.mark.benchmark
def benchmark_match(seq_code, kmer_table):
    sequence = seq.NucleotideSequence()
    sequence.code = seq_code
    kmer_table.match(sequence)


@pytest.mark.benchmark
def benchmark_match_kmer_selection(kmer_code, kmer_table):
    positions = np.arange(len(kmer_code), dtype=np.uint32)
    kmer_table.match_kmer_selection(positions, kmer_code)


@pytest.mark.benchmark
def benchmark_match_table(kmer_table):
    kmer_table.match_table(kmer_table)


@pytest.mark.benchmark
def test_pickle_and_unpickle(kmer_table):
    pickle.loads(pickle.dumps(kmer_table))


@pytest.mark.benchmark
def benchmark_score_threshold_rule(kmer_alphabet, kmer_code):
    SCORE_THRESHOLD = 10

    matrix = align.SubstitutionMatrix.std_nucleotide_matrix()
    rule = align.ScoreThresholdRule(matrix, SCORE_THRESHOLD)
    for kmer in kmer_code:
        rule.similar_kmers(kmer_alphabet, kmer)
