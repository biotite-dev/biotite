# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
import tempfile
import numpy as np
from requests.exceptions import ConnectionError
import pytest
import biotite.database.entrez as entrez
import biotite.sequence.io.fasta as fasta
from biotite.database import RequestError
from ..util import cannot_connect_to


NCBI_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/"


@pytest.mark.skipif(
    cannot_connect_to(NCBI_URL),
    reason="NCBI Entrez is not available"
)
@pytest.mark.parametrize(
    "common_name, as_file_like",
    itertools.product([False, True], [False, True])
)
def test_fetch(common_name, as_file_like):
    path = None if as_file_like else tempfile.gettempdir()
    db_name = "Protein" if common_name else "protein"
    file = entrez.fetch(
        "1L2Y_A", path, "fa", db_name, "fasta", overwrite=True
    )
    fasta_file = fasta.FastaFile.read(file)
    prot_seq = fasta.get_sequence(fasta_file)
    assert len(prot_seq) == 20

@pytest.mark.skipif(
    cannot_connect_to(NCBI_URL),
    reason="NCBI Entrez is not available"
)
@pytest.mark.parametrize("as_file_like", [False, True])
def test_fetch_single_file(as_file_like):
    if as_file_like:
        file_name = None
    else:
        file = tempfile.NamedTemporaryFile("r", suffix=".fa")
        file_name = file.name
    
    downloaded_file_name = entrez.fetch_single_file(
        ["1L2Y_A", "3O5R_A"], file_name, "protein", "fasta"
    )
    fasta_file = fasta.FastaFile.read(downloaded_file_name)
    prot_seqs = fasta.get_sequences(fasta_file)
    assert len(prot_seqs) == 2

    if not as_file_like:
        file.close()

@pytest.mark.skipif(
    cannot_connect_to(NCBI_URL),
    reason="NCBI Entrez is not available"
)
def test_fetch_invalid():
    with pytest.raises(RequestError):
        # Empty ID list
        file = entrez.fetch_single_file(
            [], None, "protein", "fasta", overwrite=True)
    with pytest.raises(RequestError):
        # Nonexisting ID
        file = entrez.fetch(
            "xxxx", None, "fa", "protein", "fasta", overwrite=True
        )