# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import tempfile
import pytest
import biotite.database.afdb as afdb
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
from biotite.database import RequestError
from tests.util import cannot_connect_to

AFDB_URL = "https://alphafold.ebi.ac.uk/"


@pytest.mark.skipif(cannot_connect_to(AFDB_URL), reason="AlphaFold DB is not available")
@pytest.mark.parametrize("as_file_like", [False, True])
@pytest.mark.parametrize("entry_id", ["P12345", "AF-P12345-F1", "AF-P12345F1"])
@pytest.mark.parametrize("format", ["pdb", "cif", "bcif"])
def test_fetch(as_file_like, entry_id, format):
    """
    Check if files in different formats can be downloaded by being able to parse them.
    Also ensure that the downloaded file refers to the given input ID
    """
    path = None if as_file_like else tempfile.gettempdir()
    file_path_or_obj = afdb.fetch(entry_id, format, path, overwrite=True)
    if format == "pdb":
        file = pdb.PDBFile.read(file_path_or_obj)
        pdb.get_structure(file)
    elif format == "cif":
        file = pdbx.CIFFile.read(file_path_or_obj)
        pdbx.get_structure(file)
        assert file.block["struct_ref"]["pdbx_db_accession"].as_item() == "P12345"
    elif format == "bcif":
        file = pdbx.BinaryCIFFile.read(file_path_or_obj)
        pdbx.get_structure(file)
        assert file.block["struct_ref"]["pdbx_db_accession"].as_item() == "P12345"


@pytest.mark.skipif(cannot_connect_to(AFDB_URL), reason="AlphaFold DB is not available")
def test_fetch_multiple():
    """
    Check if multiple files can be downloaded by being able to parse them.
    """
    ids = ["P12345", "Q8K9I1"]
    files = afdb.fetch(ids, "cif", tempfile.gettempdir(), overwrite=True)
    for file in files:
        assert "citation_author" in pdbx.CIFFile.read(file).block


@pytest.mark.skipif(cannot_connect_to(AFDB_URL), reason="AlphaFold DB is not available")
@pytest.mark.parametrize("format", ["pdb", "cif", "bcif"])
@pytest.mark.parametrize("invalid_id", ["", "XYZ", "A0A12345"])
@pytest.mark.parametrize("bypass_metadata", [False, True])
def test_fetch_invalid(monkeypatch, format, invalid_id, bypass_metadata):
    """
    Check if proper exceptions are raised if a given ID is invalid.
    Also check whether the check works on the file retrieval level via
    :func:`_get_file_url()`, by bypassing the metadata check.
    """
    import biotite.database.afdb.download as module

    if bypass_metadata:
        monkeypatch.setattr(
            module,
            "_get_file_url",
            lambda id, f: f"https://alphafold.ebi.ac.uk/files/AF-{id}-F1-model_v4.{f}",
        )
    with pytest.raises((RequestError, ValueError)):
        afdb.fetch(invalid_id, format)
