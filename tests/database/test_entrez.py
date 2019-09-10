# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite
import biotite.database.entrez as entrez
import biotite.sequence.io.fasta as fasta
import numpy as np
from requests.exceptions import ConnectionError
import pytest


@pytest.mark.xfail(raises=ConnectionError)
@pytest.mark.parametrize("common_name", [False, True])
def test_fetch(common_name):
    db_name = "Protein" if common_name else "protein"
    file = entrez.fetch("1L2Y_A", biotite.temp_dir(), "fa", db_name,
                        "fasta", overwrite=True)
    fasta_file = fasta.FastaFile()
    fasta_file.read(file)
    prot_seq = fasta.get_sequence(fasta_file)

@pytest.mark.xfail(raises=ConnectionError)
def test_fetch_single_file():
    file = entrez.fetch_single_file(
        ["1L2Y_A", "3O5R_A"], biotite.temp_file("fa"), "protein", "fasta"
    )
    fasta_file = fasta.FastaFile()
    fasta_file.read(file)
    prot_seqs = fasta.get_sequences(fasta_file)
    assert len(prot_seqs) == 2

@pytest.mark.xfail(raises=ConnectionError)
def test_fetch_invalid():
    with pytest.raises(ValueError):
        file = entrez.fetch("xxxx", biotite.temp_dir(), "fa", "protein",
                            "fasta", overwrite=True)