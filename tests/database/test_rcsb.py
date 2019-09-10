# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite
import biotite.database.rcsb as rcsb
import biotite.structure.io as strucio
import numpy as np
from requests.exceptions import ConnectionError
import pytest


@pytest.mark.xfail(raises=ConnectionError)
def test_fetch_cif():
    file = rcsb.fetch("1l2y", "cif", biotite.temp_dir(), overwrite=True)
    array = strucio.load_structure(file)


@pytest.mark.xfail(raises=ConnectionError)
def test_fetch_pdb():
    file = rcsb.fetch("1l2y", "cif", biotite.temp_dir(), overwrite=True)
    array = strucio.load_structure(file)


@pytest.mark.xfail(raises=ConnectionError)
def test_fetch_invalid():
    with pytest.raises(ValueError):
        file = rcsb.fetch("xxxx", "cif", biotite.temp_dir(), overwrite=True)


@pytest.mark.xfail(raises=ConnectionError)
def test_search():
    query1 = rcsb.ResolutionQuery(0.0, 0.8)
    query2 = rcsb.MolecularWeightQuery(0, 1000)
    ids_query1 = sorted(rcsb.search(query1))
    ids_query2 = sorted(rcsb.search(query2))
    ids_comp = sorted(rcsb.search(rcsb.CompositeQuery("or", [query1, query2])))
    ids_comp2 = []
    for id in ids_query1 + ids_query2:
        if id not in ids_comp2:
            ids_comp2.append(id)
    assert ids_comp == sorted(ids_comp2)


@pytest.mark.xfail(raises=ConnectionError)
def test_search_empty():
    ids = rcsb.search(rcsb.MolecularWeightQuery(0, 1))
    assert len(ids) == 0


@pytest.mark.xfail(raises=ConnectionError)
def test_search_invalid():
    class InvalidQuery(rcsb.SimpleQuery):
        def __init__(self):
            super().__init__("InvalidQuery", "gibberish")
            self.add_param("foo", "bar")
    with pytest.raises(ValueError):
        ids = rcsb.search(InvalidQuery())


@pytest.mark.xfail(raises=ConnectionError)
@pytest.mark.parametrize(
    # IMPORTANT NOTE: Since the PDB continuously adds new structures,
    # the expected IDs might need to be updated,
    # if an 'AssertionError' occurs
    "query_type, params, exp_ids",
    [
        (
            rcsb.ResolutionQuery,
            {"max": 0.6},
            ["1EJG", "1I0T", "3NIR", "3P4J", "5D8V", "5NW3"]
        ),
        (
            rcsb.BFactorQuery,
            {"min": 1.0, "max": 1.5},
            ["1G2K", "2OL9", "3ECO", "3INZ", "3LN2", "4CA4", "6M9I"]
        ),
        (
            rcsb.MolecularWeightQuery,
            {"min": 50000000},
            ["6CGV", "4F5X"]
        ),
        (
            rcsb.MoleculeTypeQuery,
            {"rna": False, "dna": False, "hybrid": True, "protein": False},
            60
        ),
        (
            rcsb.MethodQuery,
            {"method": "fiber diffraction", "has_data": True},
            ["1HGV", "1IFP", "1QL1", "2C0W", "2XKM",
             "2ZWH", "3HQV", "3HR2", "3PDM", "4IFM"]
        ),
        (
            rcsb.PubMedIDQuery,
            {"ids": [6726807, 10490104]},
            ["2HHB", "3HHB", "4HHB", "9GAA", "9GAC", "9GAF",]
        ),
    ]
)
def test_simple_query_types(query_type, params, exp_ids):
    query = query_type(**params)
    ids = rcsb.search(query)
    if isinstance(exp_ids, int):
        assert len(ids) == exp_ids
    else:
        assert set(ids) == set(exp_ids)