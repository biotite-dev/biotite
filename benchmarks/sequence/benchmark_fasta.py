import os
from pathlib import Path
import pytest
import biotite.sequence as seq
import biotite.sequence.io.fasta as fasta
from tests.util import data_dir


@pytest.mark.parametrize(
    ["fasta_path", "seq_type"],
    [
        # Single nucleotide sequence entry
        (Path(data_dir("sequence")) / "ec_bl21.fasta", seq.NucleotideSequence),
        # Multiple protein sequence entries
        (Path(data_dir("sequence")) / "cas9.fasta", seq.ProteinSequence),
    ],
)
@pytest.mark.benchmark
def benchmark_get_sequences(fasta_path, seq_type):
    fasta_file = fasta.FastaFile.read(fasta_path)
    fasta.get_sequences(fasta_file, seq_type)


@pytest.mark.benchmark
def benchmark_get_a3m_alignments():
    a3m_file = fasta.FastaFile.read(
        os.path.join(data_dir("sequence"), "1a00_A_uniref90.a3m")
    )
    fasta.get_a3m_alignments(a3m_file, seq_type=seq.ProteinSequence)
