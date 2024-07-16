# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
import tempfile
import pytest
import biotite.database.alphafold as alphafold
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
from biotite.database import RequestError
from ..util import cannot_connect_to


ALPHAFOLD_URL = "https://alphafold.com/"


@pytest.mark.skipif(
    cannot_connect_to(ALPHAFOLD_URL),
    reason="AlphaFold is not available"
)
@pytest.mark.parametrize(
    "as_file_like",
    itertools.product([False, True])
)
def test_fetch(as_file_like):
    path = None if as_file_like else tempfile.gettempdir()
    file = alphafold.fetch(
        "P12345", path, overwrite=True
    )
    pdb_file = pdb.PDBFile.read(file)
    array_stack = pdb_file.get_structure()
    assert len(array_stack) > 0
    

@pytest.mark.skipif(
    cannot_connect_to(ALPHAFOLD_URL),
    reason="AlphaFold is not available"
)
@pytest.mark.parametrize("format", ["pdb", "cif"])
def test_fetch_invalid(format):
    with pytest.raises(RequestError):
        file = alphafold.fetch(
            "XYZ", target_path=tempfile.gettempdir(), format=format, overwrite=True
        )


@pytest.mark.skipif(
    cannot_connect_to(ALPHAFOLD_URL),
    reason="AlphaFold is not available"
)
@pytest.mark.parametrize("format", ["pdb", "cif"])
def test_fetch_multiple(format):
        acc = ["P12345", "P12345"]
        files = alphafold.fetch(
            acc, target_path=tempfile.gettempdir(), format=format, overwrite=True
        )
        print(files)
        for file in files:
            if format == "pdb":
                pdb_file = pdb.PDBFile.read(file)
                structure = pdb_file.get_structure()
                assert len(structure) > 0 
            elif format == "cif":
                cif_file = pdbx.PDBxFile.read(file) 
                assert "citation_author" in cif_file.keys()




