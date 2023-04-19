# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
import itertools
import tempfile
import pytest
import numpy as np
import biotite.database.pubchem as pubchem
import biotite.structure.io.mol as mol
from biotite.database import RequestError
from ..util import cannot_connect_to, data_dir


PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/"


@pytest.mark.skipif(
    cannot_connect_to(PUBCHEM_URL),
    reason="Pubchem is not available"
)
@pytest.mark.parametrize(
    "format, as_file_like",
    itertools.product(["sdf", "png"], [False, True])
)
def test_fetch(format, as_file_like):
    """
    Check download of a record in binary and text form.
    """
    CID = 2244

    path = None if as_file_like else tempfile.gettempdir()
    file_path_or_obj = pubchem.fetch(2244, format, path, overwrite=True)
    if format == "sdf":
        mol_file = mol.MOLFile.read(file_path_or_obj)
        mol_name, _, _, _, _, _, _, _, _ = mol_file.get_header()
        assert int(mol_name) == CID
        # This should be a readable structure
        mol_file.get_structure()


@pytest.mark.skipif(
    cannot_connect_to(PUBCHEM_URL),
    reason="RCSB PDB is not available"
)
def test_fetch_invalid():
    """
    An exception is expected when the CID is not available.
    """
    with pytest.raises(RequestError, match="No record data for CID 1234567890"):
        pubchem.fetch(1234567890)


@pytest.mark.skipif(
    cannot_connect_to(PUBCHEM_URL),
    reason="RCSB PDB is not available"
)
@pytest.mark.parametrize(
    "query, ref_ids",
    [
        (pubchem.NameQuery("Alanine"), [5950, 449619, 7311724, 155817681]),
        (pubchem.SmilesQuery("CCCC"), [7843]),
        (pubchem.InchiQuery("InChI=1S/C4H10/c1-3-4-2/h3-4H2,1-2H3"), [7843]),
        (pubchem.InchiKeyQuery("IJDNQMDRQITEOD-UHFFFAOYSA-N"), [7843]),
    ]
)
def test_search_simple(query, ref_ids):
    """
    Checks for the simpler `Query` types, where the output is known.
    """
    assert set(pubchem.search(query)) == set(ref_ids)


@pytest.mark.skipif(
    cannot_connect_to(PUBCHEM_URL),
    reason="RCSB PDB is not available"
)
@pytest.mark.parametrize("number", [None, 10])
def test_search_formula(number):
    """
    Download a structure and search for its molecular formula in
    PubChem.
    Expect that the original structure is among these results.
    """
    CID = 101608985

    atoms = mol.MOLFile.read(pubchem.fetch(CID)).get_structure()
    test_cids = pubchem.search(
        pubchem.FormulaQuery.from_atoms(atoms, number=number)
    )

    assert CID in (test_cids)
    if number is not None:
        # This request would normally give more than 10 results
        assert len(test_cids) == number