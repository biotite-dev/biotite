# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
import tempfile
import pytest
import biotite.database.uniprot as uniprot
import biotite.sequence.io.fasta as fasta
from biotite.database import RequestError
from ..util import cannot_connect_to


UNIPROT_URL = "https://www.uniprot.org/"


@pytest.mark.skipif(
    cannot_connect_to(UNIPROT_URL),
    reason="UniProt is not available"
)
@pytest.mark.parametrize(
    "as_file_like",
    itertools.product([False, True])
)
def test_fetch(as_file_like):
    path = None if as_file_like else tempfile.gettempdir()

    # UniProtKB
    file = uniprot.fetch(
        "P12345", "fasta", path, overwrite=True
    )
    fasta_file = fasta.FastaFile.read(file)
    prot_seq = fasta.get_sequence(fasta_file)
    assert len(prot_seq) == 430

    # UniRef
    file = uniprot.fetch(
        "UniRef90_P99999", "fasta", path, overwrite=True
    )
    fasta_file = fasta.FastaFile.read(file)
    prot_seq = fasta.get_sequence(fasta_file)
    assert len(prot_seq) == 105

    # UniParc
    file = uniprot.fetch(
        "UPI000000001F", "fasta", path, overwrite=True
    )
    fasta_file = fasta.FastaFile.read(file)
    prot_seq = fasta.get_sequence(fasta_file)
    assert len(prot_seq) == 551


@pytest.mark.skipif(
    cannot_connect_to(UNIPROT_URL),
    reason="UniProt is not available"
)
@pytest.mark.parametrize("format", ["fasta", "gff", "txt", "xml", "rdf", "tab"])
def test_fetch_invalid(format):
    with pytest.raises(RequestError):
        file = uniprot.fetch(
            "xxxx", format, tempfile.gettempdir(), overwrite=True
        )


@pytest.mark.skipif(
    cannot_connect_to(UNIPROT_URL),
    reason="UniProt is not available"
)
def test_search_simple():
    query = uniprot.SimpleQuery("accession", "P12345")
    assert uniprot.search(query) \
        == ['P12345']


@pytest.mark.skipif(
    cannot_connect_to(UNIPROT_URL),
    reason="UniProt is not available"
)
def test_search_composite():
    query = uniprot.SimpleQuery("accession", "P12345") & uniprot.SimpleQuery("reviewed", "true")
    assert uniprot.search(query) \
        == ['P12345']

