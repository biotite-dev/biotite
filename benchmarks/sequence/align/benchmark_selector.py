import numpy as np
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align

SEQ_LENGTH = 10_000
K = 8
S = 4
WINDOW = 10
ALPHABET = seq.NucleotideSequence.alphabet_unamb
KMER_ALPHABET = align.KmerAlphabet(ALPHABET, K)


@pytest.fixture(scope="module")
def sequence():
    np.random.seed(0)
    s = seq.NucleotideSequence()
    s.code = np.random.randint(len(ALPHABET), size=SEQ_LENGTH)
    return s


@pytest.mark.parametrize(
    "selector",
    [
        align.MinimizerSelector(KMER_ALPHABET, window=WINDOW),
        align.SyncmerSelector(ALPHABET, K, S),
        align.CachedSyncmerSelector(ALPHABET, K, S),
        align.MincodeSelector(KMER_ALPHABET, compression=4),
    ],
    ids=lambda x: x.__class__.__name__,
)
def benchmark_select(sequence, selector):
    """
    Select k-mers from a sequence using different selection strategies.
    """
    selector.select(sequence)
