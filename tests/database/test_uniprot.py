# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
import tempfile
import pytest
import biotite.database.uniprot as uniprot
import biotite.sequence.io.fasta as fasta
from ..util import cannot_connect_to


UNIPROT_URL = "https://www.uniprot.org/"


pytest.mark.skipif(
    cannot_connect_to(UNIPROT_URL),
    reason="UniProtKB is not available"
)


@pytest.mark.parametrize(
    "common_name, as_file_like",
    itertools.product([False, True], [False, True])
)
def test_fetch(common_name, as_file_like):
    path = None if as_file_like else tempfile.gettempdir()
    db_name = "UniProtKB" if common_name else "uniprot"
    file = uniprot.fetch(
        "P12345", db_name, "fasta", path, overwrite=True
    )
    fasta_file = fasta.FastaFile.read(file)
    prot_seq = fasta.get_sequence(fasta_file)
    assert len(prot_seq) == 430