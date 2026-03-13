from functools import partial
from pathlib import Path
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align
import biotite.sequence.io.fasta as fasta
from tests.util import data_dir

GAP_PENALTY = (-10, -1)


@pytest.fixture(scope="module")
def sequences():
    fasta_file = fasta.FastaFile.read(Path(data_dir("sequence")) / "cas9.fasta")
    return [seq.ProteinSequence(s) for s in fasta_file.values()]


@pytest.fixture(scope="module")
def matrix():
    return align.SubstitutionMatrix.std_protein_matrix()


@pytest.fixture(scope="module")
def seq_pair(sequences):
    return sequences[0], sequences[1]


@pytest.fixture(scope="module")
def seed(seq_pair):
    return (len(seq_pair[0]) // 2, len(seq_pair[1]) // 2)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "method",
    [
        partial(align.align_optimal, gap_penalty=GAP_PENALTY),
        partial(align.align_banded, band=(-50, 50), gap_penalty=GAP_PENALTY),
        partial(align.align_local_gapped, threshold=100, gap_penalty=GAP_PENALTY),
    ],
    ids=lambda x: x.func.__name__,
)
def benchmark_align_pairwise(seq_pair, matrix, seed, method):
    """
    Perform pairwise sequence alignment using different algorithms.
    """
    kwargs = {"seed": seed} if method.func is align.align_local_gapped else {}
    method(seq_pair[0], seq_pair[1], matrix, **kwargs)
