# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import datetime
import itertools
import numpy as np
from requests.exceptions import ConnectionError
import pytest
import biotite
import biotite.database.rcsb as rcsb
import biotite.structure.io as strucio
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.mmtf as mmtf
from biotite.database import RequestError


@pytest.mark.xfail(raises=ConnectionError)
@pytest.mark.parametrize(
    "format, as_file_like",
    itertools.product(["pdb", "cif", "mmtf"], [False, True])
)
def test_fetch(format, as_file_like):
    path = None if as_file_like else biotite.temp_dir()
    file_path_or_obj = rcsb.fetch("1l2y", format, path, overwrite=True)
    if format == "pdb":
        file = pdb.PDBFile()
        file.read(file_path_or_obj)
        pdb.get_structure(file)
    elif format == "pdbx":
        file = pdbx.PDBxFile()
        file.read(file_path_or_obj)
        pdbx.get_structure(file)
    elif format == "mmtf":
        file = mmtf.MMTFFile()
        file.read(file_path_or_obj)
        mmtf.get_structure(file)


@pytest.mark.xfail(raises=ConnectionError)
def test_fetch_invalid():
    with pytest.raises(RequestError):
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
    with pytest.raises(RequestError):
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
        #(
        #    rcsb.ChainCountQuery,
        #    {"min": 100, "max": 101},
        #    []
        #),
        (
            rcsb.ChainCountQuery,
            {"min": 100, "max": 101, "bio_assembly": True},
            ["6NUT", "5HGE", "3J2W", "1Z8Y"]
        ),
        (
            rcsb.EntityCountQuery,
            {"min": 85, "max": 100, "entity_type": "protein"},
            ["6HIX", "5LYB"]
        ),
        (
            rcsb.ModelCountQuery,
            {"min": 60, "max": 61},
            ["1BBO", "1GB1", "1O5P", "1XU6", "2LUM", "2NO8"]
        ),
        (
            rcsb.ChainLengthQuery,
            {"min": 1000, "max": 1000},
            ["3DEC", "3W5B", "4NAB", "5UM6", "5ZMV", "5ZMW"]
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
            rcsb.SoftwareQuery,
            {"name": "Gromacs"},
            ["3K2S"]
        ),
        (
            rcsb.PubMedIDQuery,
            {"ids": ["6726807", "10490104"]},
            ["2HHB", "3HHB", "4HHB", "9GAA", "9GAC", "9GAF",]
        ),
        (
            rcsb.UniProtIDQuery,
            {"ids": ["P69905"]},
            263
        ),
        (
            rcsb.PfamIDQuery,
            {"ids": ["PF07388"]},
            ["5WC6", "5WC8", "5WCN", "5WD7"]
        ),
        (
            rcsb.SequenceClusterQuery,
            {"cluster_id": "5000"},
            ["1WFD", "1XRI", "2Q47", "2RNO"]
        ),
        (
            rcsb.TextSearchQuery,
            {"text": "Miniprotein Construct TC5b"},
            ["1L2Y"]
        ),
        (
            rcsb.KeywordQuery,
            {"keyword": "ION CHANNEL INHIBITOR"},
            ["2CK4", "2CK5"]
        ),
        (
            rcsb.TitleQuery,
            {"text": "tc5b"},
            ["1L2Y"]
        ),
        (
            rcsb.DecriptionQuery,
            {"text": "tc5b"},
            ["1L2Y"]
        ),
        (
            rcsb.MacromoleculeNameQuery,
            {"name": "tc5b"},
            ["1L2Y"]
        ),
        (
            rcsb.ExpressionOrganismQuery,
            {"name": "Bacillus subtilis"},
            222
        ),
        (
            rcsb.AuthorQuery,
            {"name": "Neidigh, J.W."},
            ["1JRJ", "1L2Y", "2JOF", "2O3P", "2O63", "2O64", "2O65"]
        ),
        (
            rcsb.AuthorQuery,
            {"name": "Neidigh, J.W.", "exact": True},
            ["1JRJ", "1L2Y", "2JOF", "2O3P", "2O63", "2O64", "2O65"]
        ),
        (
            rcsb.AuthorQuery,
            {"name": "Neidigh, J.W.", "exact": True},
            ["1JRJ", "1L2Y", "2JOF", "2O3P", "2O63", "2O64", "2O65"]
        ),
        (
            rcsb.DateQuery,
            {
                "min_date": datetime.date(2008, 8, 1 ),
                "max_date": datetime.date(2008, 8, 30),
                "event": "deposition"
            },
            550
        ),
        (
            rcsb.DateQuery,
            {
                "min_date": datetime.date(2008, 8, 1 ),
                "max_date": datetime.date(2008, 8, 30),
                "event": "release"
            },
            566
        ),
        (
            rcsb.DateQuery,
            {
                "min_date": "2008-08-01",
                "max_date": "2008-09-30",
                "event": "revision"
            },
            2
        ),
    ]
)
def test_simple_query_types(query_type, params, exp_ids):
    query = query_type(**params)
    print("Query:")
    print(query)
    ids = rcsb.search(query)
    if isinstance(exp_ids, int):
        assert len(ids) == exp_ids
    else:
        assert set(ids) == set(exp_ids)