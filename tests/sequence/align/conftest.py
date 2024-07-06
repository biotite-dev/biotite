# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
import pytest
import biotite.sequence as seq
import biotite.sequence.io.fasta as fasta
from ...util import data_dir


@pytest.fixture
def sequences():
    """
    10 Cas9 sequences.
    """
    fasta_file = fasta.FastaFile.read(join(data_dir("sequence"), "cas9.fasta"))
    return [seq.ProteinSequence(sequence) for sequence in fasta_file.values()]