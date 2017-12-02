# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License. Please see 'LICENSE.rst' for further information.

import biotite
import biotite.database.rcsb as rcsb
import biotite.structure.io as strucio
import numpy as np
from requests.exceptions import ConnectionError
import pytest


@pytest.mark.xfail(raises=ConnectionError)
def test_fetch_cif():
    file = rcsb.fetch("1l2y", "cif", biotite.temp_dir(), overwrite=True)
    array = strucio.get_structure_from(file)

@pytest.mark.xfail(raises=ConnectionError)
def test_fetch_pdb():
    file = rcsb.fetch("1l2y", "cif", biotite.temp_dir(), overwrite=True)
    array = strucio.get_structure_from(file)

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

