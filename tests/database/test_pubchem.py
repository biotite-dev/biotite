# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import functools
import itertools
import re
import tempfile
import numpy as np
import pytest
import biotite.database.pubchem as pubchem
import biotite.structure.io.mol as mol
from biotite.database import RequestError
from tests.util import cannot_connect_to

PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/"


def accept_busy_pubchem(func):
    """
    Accept that PubChem is currently busy. and the test fails for this reason.

    Although the
    `Dynamic Request Throttling <https://pubchem.ncbi.nlm.nih.gov/docs/dynamic-request-throttling>`_
    is implemented in :mod:`biotite.database.pubchem`, it does not work properly in the
    CI.

    This decorator must be added before any other decorator from ``pytest``
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RequestError as e:
            if str(e) == "Too many requests or server too busy":
                pytest.skip(str(e))
            else:
                raise e

    return wrapper


@pytest.mark.skipif(cannot_connect_to(PUBCHEM_URL), reason="Pubchem is not available")
@pytest.mark.parametrize(
    "format, as_file_like", itertools.product(["sdf", "png"], [False, True])
)
@accept_busy_pubchem
def test_fetch(format, as_file_like):
    """
    Check download of a record in binary and text form.
    """
    CID = 2244

    path = None if as_file_like else tempfile.gettempdir()
    file_path_or_obj = pubchem.fetch(CID, format, path, overwrite=True)
    if format == "sdf":
        mol_file = mol.MOLFile.read(file_path_or_obj)
        assert int(mol_file.header.mol_name) == CID
        # This should be a readable structure
        mol_file.get_structure()


@pytest.mark.skipif(cannot_connect_to(PUBCHEM_URL), reason="PubChem is not available")
@pytest.mark.parametrize("as_structural_formula", [False, True])
@accept_busy_pubchem
def test_fetch_structural_formula(as_structural_formula):
    """
    Check download of structure as structural formula and 3D conformer.
    The 3D conformer should expand into the z-dimension while the
    structural formula must not.
    """
    CID = 2244

    mol_file = mol.MOLFile.read(
        pubchem.fetch(CID, as_structural_formula=as_structural_formula)
    )
    atoms = mol_file.get_structure()

    if as_structural_formula:
        assert np.all(atoms.coord[:, 2] == 0)
    else:
        assert np.any(atoms.coord[:, 2] != 0)


@pytest.mark.skipif(cannot_connect_to(PUBCHEM_URL), reason="PubChem is not available")
@accept_busy_pubchem
def test_fetch_invalid():
    """
    An exception is expected when the CID is not available.
    """
    with pytest.raises(
        RequestError, match=re.escape("No records found for the given CID(s)")
    ):
        pubchem.fetch(1234567890)


@pytest.mark.skipif(cannot_connect_to(PUBCHEM_URL), reason="PubChem is not available")
@pytest.mark.parametrize(
    "query, ref_ids",
    [
        (pubchem.NameQuery("Alanine"), [5950]),
        (pubchem.SmilesQuery("CCCC"), [7843]),
        (pubchem.InchiQuery("InChI=1S/C4H10/c1-3-4-2/h3-4H2,1-2H3"), [7843]),
        (pubchem.InchiKeyQuery("IJDNQMDRQITEOD-UHFFFAOYSA-N"), [7843]),
    ],
    ids=["NameQuery", "SmilesQuery", "InchiQuery", "InchiKeyQuery"],
)
@accept_busy_pubchem
def test_search_simple(query, ref_ids):
    """
    Checks for the simpler `Query` types, where the output is known.
    """
    # NOTE: The full set of ref ids returned by pubchem routinely changes.
    #       For this reason, we check that the expected ref_ids set is
    #       a subset of the actual set. The expected ref_ids set should
    #       be stable over time.
    assert set(ref_ids).issubset(pubchem.search(query))


@pytest.mark.skipif(cannot_connect_to(PUBCHEM_URL), reason="PubChem is not available")
@accept_busy_pubchem
def test_search_formula():
    """
    Download a structure and search for its molecular formula in
    PubChem.
    Expect that the original structure is among these results.
    """
    CID = 101608985

    atoms = mol.MOLFile.read(pubchem.fetch(CID)).get_structure()
    test_cids = pubchem.search(pubchem.FormulaQuery.from_atoms(atoms))

    assert CID in (test_cids)


@pytest.mark.skipif(cannot_connect_to(PUBCHEM_URL), reason="PubChem is not available")
@pytest.mark.parametrize(
    "cid, from_atoms, query_type",
    itertools.product(
        [2244], [False, True], [pubchem.SuperstructureQuery, pubchem.SubstructureQuery]
    ),
)
@accept_busy_pubchem
def test_search_super_and_substructure(cid, from_atoms, query_type):
    """
    Super- and substructure searches should return structures with less
    or more atoms than the input structure, respectively.
    """
    NUMBER = 5

    original_atoms = mol.MOLFile.read(pubchem.fetch(cid)).get_structure()

    if from_atoms:
        query = query_type.from_atoms(original_atoms, number=NUMBER)
    else:
        query = query_type(cid=cid, number=NUMBER)
    cids = pubchem.search(query)

    # Expect number of returned CIDs to be limited by given max number
    assert len(cids) == NUMBER
    if query_type == pubchem.SubstructureQuery:
        # Expect that the input itself is the top hit
        assert cid in cids

    for result_cid in cids:
        atoms = mol.MOLFile.read(
            # The compound might only be available as structural formula
            pubchem.fetch(result_cid, as_structural_formula=True)
        ).get_structure()
        if query_type == pubchem.SuperstructureQuery:
            # Expect that the input is the superstructure
            # of the query result
            # Simple test based of number of atoms
            assert atoms.array_length() <= original_atoms.array_length()
        else:
            # Expect that the input is the substructure
            # of the query result
            assert atoms.array_length() >= original_atoms.array_length()


@pytest.mark.skipif(cannot_connect_to(PUBCHEM_URL), reason="PubChem is not available")
@pytest.mark.parametrize(
    "conformation_based, from_atoms", itertools.product([False, True], [False, True])
)
@accept_busy_pubchem
def test_search_similarity(conformation_based, from_atoms):
    """
    The input structure should have a similarity of 1.0 to itself.
    Since different isotopes have the same *Tanimoto*, results also
    include other compounds.
    """
    CID = 2244

    if from_atoms:
        original_atoms = mol.MOLFile.read(pubchem.fetch(CID)).get_structure()
        query = pubchem.SimilarityQuery.from_atoms(
            original_atoms, threshold=1.0, conformation_based=conformation_based
        )
    else:
        query = pubchem.SimilarityQuery(
            cid=CID, threshold=1.0, conformation_based=conformation_based
        )
    cids = pubchem.search(query)

    assert CID in cids


@pytest.mark.skipif(cannot_connect_to(PUBCHEM_URL), reason="PubChem is not available")
@pytest.mark.parametrize("from_atoms", [False, True])
@accept_busy_pubchem
def test_search_identity(from_atoms):
    """
    The input structure should be identical to itself.
    """
    CID = 2244

    if from_atoms:
        original_atoms = mol.MOLFile.read(pubchem.fetch(CID)).get_structure()
        query = pubchem.IdentityQuery.from_atoms(original_atoms)
    else:
        query = pubchem.IdentityQuery(cid=CID)
    cids = pubchem.search(query)

    assert cids == [CID]
