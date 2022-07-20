# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
import itertools
import tempfile
import pytest
import numpy as np
import biotite.database.rcsb as rcsb
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.mmtf as mmtf
import biotite.sequence.io.fasta as fasta
import biotite.sequence.align as align
from biotite.database import RequestError
from ..util import cannot_connect_to, data_dir


RCSB_URL = "https://www.rcsb.org/"


@pytest.mark.skipif(
    cannot_connect_to(RCSB_URL),
    reason="RCSB PDB is not available"
)
@pytest.mark.parametrize(
    "format, as_file_like",
    itertools.product(["pdb", "cif", "mmtf", "fasta"], [False, True])
)
def test_fetch(format, as_file_like):
    path = None if as_file_like else tempfile.gettempdir()
    file_path_or_obj = rcsb.fetch("1l2y", format, path, overwrite=True)
    if format == "pdb":
        file = pdb.PDBFile.read(file_path_or_obj)
        pdb.get_structure(file)
    elif format == "pdbx":
        file = pdbx.PDBxFile.read(file_path_or_obj)
        pdbx.get_structure(file)
    elif format == "mmtf":
        file = mmtf.MMTFFile.read(file_path_or_obj)
        mmtf.get_structure(file)
    elif format == "fasta":
        file = fasta.FastaFile.read(file_path_or_obj)
        # Test if the file contains any sequences
        assert len(fasta.get_sequences(file)) > 0


@pytest.mark.skipif(
    cannot_connect_to(RCSB_URL),
    reason="RCSB PDB is not available"
)
@pytest.mark.parametrize("format", ["pdb", "cif", "mmtf", "fasta"])
def test_fetch_invalid(format):
    with pytest.raises(RequestError):
        file = rcsb.fetch(
            "xxxx", format, tempfile.gettempdir(), overwrite=True
        )


def test_search_basic():
    query = rcsb.BasicQuery("tc5b")
    assert rcsb.search(query) == ["1L2Y"]
    assert rcsb.count(query) == 1


@pytest.mark.parametrize(
    "field, molecular_definition, params, ref_ids",
    [
        (
            "pdbx_serial_crystallography_sample_delivery_injection.preparation",
            False,
            {},
            ["6IG7", "6IG6", "7JRI", "7JR5"]
        ),
        (
            "audit_author.name",
            False,
            {"is_in": ["Neidigh, J.W."]},
            ["1JRJ", "1L2Y", "2O3P", "2O63", "2O64", "2O65"]
        ),
        (
            "rcsb_entity_source_organism.rcsb_gene_name.value",
            False,
            {"exact_match": "lacA"},
            ["5JUV", "1KQA", "1KRV", "1KRU", "1KRR", "1TG7", "1XC6", "3U7V",
             "4IUG", "4LFK", "4LFL", "4LFM", "4LFN", "5IFP", "5IFT", "5IHR",
             "4DUW", "5MGD", "5MGC"]
        ),
        (
            "struct.title",
            False,
            {"contains_words": "tc5b"},
            ["1L2Y"]
        ),
        (
            "reflns.d_resolution_high",
            False,
            {"less_or_equal": 0.6},
            ["1EJG", "1I0T", "3NIR", "3P4J", "5D8V", "5NW3", "4JLJ", "2GLT",
             "7ATG"]
        ),
        (
            "rcsb_entry_info.deposited_model_count",
            False,
            {"range_closed": (60, 61)},
            ["1BBO", "1GB1", "1O5P", "1XU6", "2LUM", "2NO8"]
        ),
        (
            "rcsb_id",
            True,
            {"exact_match": "AIN"},
            ["1OXR", "1TGM", "3IAZ", "3GCL", "6MQF", "2QQT", "4NSB"]
        ),
    ]
)
@pytest.mark.skipif(
    cannot_connect_to(RCSB_URL),
    reason="RCSB PDB is not available"
)
def test_search_field(field, molecular_definition, params, ref_ids):
    query = rcsb.FieldQuery(
        field, molecular_definition, **params
    )
    test_ids = rcsb.search(query)
    test_count = rcsb.count(query)

    assert set(test_ids) == set(ref_ids)
    assert test_count == len(ref_ids)


@pytest.mark.skipif(
    cannot_connect_to(RCSB_URL),
    reason="RCSB PDB is not available"
)
def test_search_sequence():
    IDENTIY_CUTOFF = 0.9
    pdbx_file = pdbx.PDBxFile.read(join(data_dir("structure"), "1l2y.cif"))
    ref_sequence = pdbx.get_sequence(pdbx_file)[0]
    query = rcsb.SequenceQuery(
        ref_sequence, "protein", min_identity=IDENTIY_CUTOFF
    )
    test_ids = rcsb.search(query)

    for id in test_ids:
        fasta_file = fasta.FastaFile.read(rcsb.fetch(id, "fasta"))
        test_sequence = fasta.get_sequence(fasta_file)
        matrix = align.SubstitutionMatrix.std_protein_matrix()
        alignment = align.align_optimal(
            ref_sequence, test_sequence, matrix, terminal_penalty=False
        )[0]
        identity = align.get_sequence_identity(alignment, mode="shortest")
        assert identity >= IDENTIY_CUTOFF


@pytest.mark.skipif(
    cannot_connect_to(RCSB_URL),
    reason="RCSB PDB is not available"
)
def test_search_structure():
    query = rcsb.StructureQuery("1L2Y", chain="A")
    test_ids = rcsb.search(query)
    assert "1L2Y" in test_ids


@pytest.mark.skipif(
    cannot_connect_to(RCSB_URL),
    reason="RCSB PDB is not available"
)
def test_search_motif():
    # motif is taken from official RCSB search API tutorial
    MOTIF = "C-x(2,4)-C-x(3)-[LIVMFYWC]-x(8)-H-x(3,5)-H."
    query = rcsb.MotifQuery(MOTIF, "prosite", "protein")
    test_count = rcsb.count(query)
    assert test_count == pytest.approx(503, rel=0.1)


@pytest.mark.skipif(
    cannot_connect_to(RCSB_URL),
    reason="RCSB PDB is not available"
)
def test_search_composite():
    query1 = rcsb.FieldQuery(
        "rcsb_entity_host_organism.scientific_name",
        exact_match="Homo sapiens"
    )
    query2 = rcsb.FieldQuery(
        "exptl.method",
        exact_match="SOLUTION NMR"
    )
    ids_1 = set(rcsb.search(query1))
    ids_2 = set(rcsb.search(query2))
    ids_or = set(rcsb.search(query1 | query2))
    ids_and = set(rcsb.search(query1 & query2))

    assert ids_or  == ids_1 | ids_2
    assert ids_and == ids_1 & ids_2


@pytest.mark.parametrize(
    "return_type, expected",
    [
        ("entry",              ["1L2Y"]  ),
        ("assembly",           ["1L2Y-1"]),
        ("polymer_entity",     ["1L2Y_1"]),
        ("non_polymer_entity", []        ),
        ("polymer_instance",   ["1L2Y.A"]),
    ]
)
@pytest.mark.skipif(
    cannot_connect_to(RCSB_URL),
    reason="RCSB PDB is not available"
)
def test_search_return_type(return_type, expected):
    query = rcsb.BasicQuery("tc5b")
    assert rcsb.search(query, return_type) == expected
    assert rcsb.count(query, return_type) == len(expected)


@pytest.mark.skipif(
    cannot_connect_to(RCSB_URL),
    reason="RCSB PDB is not available"
)
@pytest.mark.parametrize("seed", np.arange(5))
def test_search_range(seed):
    query = rcsb.FieldQuery(
        "rcsb_entity_host_organism.scientific_name",
        exact_match="Homo sapiens"
    )
    count = rcsb.count(query)
    ref_entries = rcsb.search(query)
    assert len(ref_entries) == count

    np.random.seed(seed)
    range = sorted(np.random.choice(count, 2, replace=False))
    test_entries = rcsb.search(query, range=range)

    assert test_entries == ref_entries[range[0] : range[1]]


@pytest.mark.skipif(
    cannot_connect_to(RCSB_URL),
    reason="RCSB PDB is not available"
)
def test_search_sort():
    query = rcsb.FieldQuery(
        "rcsb_entity_host_organism.scientific_name",
        exact_match="Homo sapiens"
    )
    entries = rcsb.search(query, sort_by="reflns.d_resolution_high")
    
    resolutions = []
    for pdb_id in entries[:5]:
        pdbx_file = pdbx.PDBxFile.read(rcsb.fetch(pdb_id, "pdbx"))
        resolutions.append(float(pdbx_file["reflns"]["d_resolution_high"]))
    
    # Check if values are sorted in descending order
    assert resolutions == list(reversed(sorted(resolutions)))


@pytest.mark.skipif(
    cannot_connect_to(RCSB_URL),
    reason="RCSB PDB is not available"
)
def test_search_empty():
    query = rcsb.BasicQuery("This will not match any ID")
    assert rcsb.search(query) == []
    assert rcsb.count(query) == 0


@pytest.mark.parametrize(
    "field, params",
    [
        (
            "invalid.field",
            {"exact_match": "Some Value"}
        ),
        (
            "exptl.method",
            {"less": 5}
        )
    ]
)
@pytest.mark.skipif(
    cannot_connect_to(RCSB_URL),
    reason="RCSB PDB is not available"
)
def test_search_invalid(field, params):
    invalid_query = rcsb.FieldQuery(field, **params)
    with pytest.raises(RequestError, match="400"):
        rcsb.search(invalid_query)
    with pytest.raises(RequestError, match="400"):
        rcsb.count(invalid_query)
