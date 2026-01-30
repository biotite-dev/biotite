from pathlib import Path
import pytest
import biotite.sequence.align as align
import biotite.sequence.io.fasta as fasta
from tests.util import data_dir


@pytest.fixture(scope="module")
def sequences():
    fasta_file = fasta.FastaFile.read(Path(data_dir("sequence")) / "cas9.fasta")
    return list(fasta.get_sequences(fasta_file).values())


@pytest.fixture(scope="module")
def matrix():
    return align.SubstitutionMatrix.std_protein_matrix()


@pytest.mark.benchmark
def benchmark_align_multiple(sequences, matrix):
    """
    Perform progressive multiple sequence alignment.
    """
    align.align_multiple(sequences, matrix, gap_penalty=(-10, -1))
