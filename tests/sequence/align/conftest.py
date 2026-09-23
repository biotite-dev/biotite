import pytest
from tests.util import data_dir


@pytest.fixture
def sequences():
    """
    10 Cas9 sequences.
    """
    # Import in function to avoid 'ModuleNotFoundError',
    # if the Rust extension is not compiled yet
    import biotite.sequence as seq
    import biotite.sequence.io.fasta as fasta

    fasta_file = fasta.FastaFile.read(data_dir("sequence") / "cas9.fasta")
    return [seq.ProteinSequence(sequence) for sequence in fasta_file.values()]
