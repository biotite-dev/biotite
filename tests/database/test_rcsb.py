# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
import tempfile
from datetime import date
from os.path import join
import numpy as np
import pytest
import biotite.database.rcsb as rcsb
import biotite.sequence.align as align
import biotite.sequence.io.fasta as fasta
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
from biotite.database import RequestError
from tests.util import cannot_connect_to, data_dir

RCSB_URL = "https://www.rcsb.org/"
# To keep RCSB search results constant over time only search for entries below this date
CUTOFF_DATE = date(2025, 4, 30)
# Search term that should only find the entry 1L2Y
TC5B_TERM = "Miniprotein Construct TC5b"


@pytest.mark.skipif(cannot_connect_to(RCSB_URL), reason="RCSB PDB is not available")
@pytest.mark.parametrize(
    "format, as_file_like",
    itertools.product(["pdb", "cif", "bcif", "fasta"], [False, True]),
)
def test_fetch(format, as_file_like):
    path = None if as_file_like else tempfile.gettempdir()
    file_path_or_obj = rcsb.fetch("1l2y", format, path, overwrite=True)
    if format == "pdb":
        file = pdb.PDBFile.read(file_path_or_obj)
        pdb.get_structure(file)
    elif format == "cif":
        file = pdbx.CIFFile.read(file_path_or_obj)
        pdbx.get_structure(file)
    elif format == "bcif":
        file = pdbx.BinaryCIFFile.read(file_path_or_obj)
        pdbx.get_structure(file)
    elif format == "fasta":
        file = fasta.FastaFile.read(file_path_or_obj)
        # Test if the file contains any sequences
        assert len(fasta.get_sequences(file)) > 0


@pytest.mark.skipif(cannot_connect_to(RCSB_URL), reason="RCSB PDB is not available")
@pytest.mark.parametrize("format", ["pdb", "cif", "bcif", "fasta"])
def test_fetch_invalid(format):
    with pytest.raises(RequestError):
        rcsb.fetch("xxxx", format, tempfile.gettempdir(), overwrite=True)


def test_search_basic():
    query = rcsb.BasicQuery(TC5B_TERM)
    assert rcsb.search(query) == ["1L2Y"]
    assert rcsb.count(query) == 1


@pytest.mark.parametrize(
    "field, molecular_definition, params, ref_ids",
    [
        (
            "pdbx_serial_crystallography_sample_delivery_injection.preparation",
            False,
            {},
            [
                "6IG7",
                "6IG6",
                "7JRI",
                "7JR5",
                "7QX4",
                "7QX5",
                "7QX6",
                "7QX7",
                "8A2O",
                "8A2P",
            ],
        ),
        (
            "audit_author.name",
            False,
            {"is_in": ["Neidigh, J.W."]},
            ["1JRJ", "1L2Y", "2O3P", "2O63", "2O64", "2O65"],
        ),
        (
            "rcsb_entity_source_organism.rcsb_gene_name.value",
            False,
            {"exact_match": "lacA"},
            [
                "5JUV",
                "1KQA",
                "1KRV",
                "1KRU",
                "1KRR",
                "3U7V",
                "4IUG",
                "4LFK",
                "4LFL",
                "4LFM",
                "4LFN",
                "5IFP",
                "5IFT",
                "5IHR",
                "4DUW",
                "5MGD",
                "5MGC",
            ],
        ),
        (
            "struct.title",
            False,
            {"contains_phrase": "Trp-Cage Miniprotein"},
            ["1L2Y"],
        ),
        (
            "reflns.d_resolution_high",
            False,
            {"less_or_equal": 0.6},
            ["1EJG", "1I0T", "3NIR", "3P4J", "5D8V", "5NW3", "4JLJ", "7ATG", "7R0H"],
        ),
        (
            "rcsb_entry_info.deposited_model_count",
            False,
            {"range_closed": (60, 61)},
            ["1BBO", "1GB1", "1O5P", "1XU6", "2LUM", "2NO8"],
        ),
        (
            "rcsb_id",
            True,
            {"exact_match": "AIN"},
            ["1OXR", "1TGM", "3IAZ", "3GCL", "6MQF", "2QQT", "4NSB", "8J3W"],
        ),
    ],
)
@pytest.mark.skipif(cannot_connect_to(RCSB_URL), reason="RCSB PDB is not available")
def test_search_field(field, molecular_definition, params, ref_ids):
    query = rcsb.FieldQuery(field, molecular_definition, **params)
    query &= rcsb.FieldQuery(
        "rcsb_accession_info.initial_release_date",
        less_or_equal=CUTOFF_DATE.isoformat(),
    )
    test_ids = rcsb.search(query)
    test_count = rcsb.count(query)

    assert set(test_ids) == set(ref_ids)
    assert test_count == len(ref_ids)


@pytest.mark.skipif(cannot_connect_to(RCSB_URL), reason="RCSB PDB is not available")
def test_search_sequence():
    IDENTIY_CUTOFF = 0.9
    pdbx_file = pdbx.BinaryCIFFile.read(join(data_dir("structure"), "1l2y.bcif"))
    ref_sequence = pdbx.get_sequence(pdbx_file)["A"]
    query = rcsb.SequenceQuery(ref_sequence, "protein", min_identity=IDENTIY_CUTOFF)
    test_ids = rcsb.search(query)
    assert len(test_ids) >= 2

    for id in test_ids:
        fasta_file = fasta.FastaFile.read(rcsb.fetch(id, "fasta"))
        test_sequence = fasta.get_sequence(fasta_file)
        matrix = align.SubstitutionMatrix.std_protein_matrix()
        alignment = align.align_optimal(
            ref_sequence, test_sequence, matrix, terminal_penalty=False
        )[0]
        identity = align.get_sequence_identity(alignment, mode="shortest")
        assert identity >= IDENTIY_CUTOFF


@pytest.mark.skipif(cannot_connect_to(RCSB_URL), reason="RCSB PDB is not available")
def test_search_structure():
    query = rcsb.StructureQuery("1L2Y", chain="A")
    test_ids = rcsb.search(query)
    assert "1L2Y" in test_ids


@pytest.mark.skipif(cannot_connect_to(RCSB_URL), reason="RCSB PDB is not available")
def test_search_motif():
    # motif is taken from official RCSB search API tutorial
    MOTIF = "C-x(2,4)-C-x(3)-[LIVMFYWC]-x(8)-H-x(3,5)-H."
    query = rcsb.MotifQuery(MOTIF, "prosite", "protein")
    test_count = rcsb.count(query, return_type="polymer_entity")
    assert test_count == pytest.approx(719, rel=0.1)


@pytest.mark.skipif(cannot_connect_to(RCSB_URL), reason="RCSB PDB is not available")
def test_search_composite():
    query1 = rcsb.FieldQuery(
        "rcsb_entity_host_organism.scientific_name", exact_match="Homo sapiens"
    )
    query2 = rcsb.FieldQuery("exptl.method", exact_match="SOLUTION NMR")
    ids_1 = set(rcsb.search(query1))
    ids_2 = set(rcsb.search(query2))
    ids_or = set(rcsb.search(query1 | query2))
    ids_and = set(rcsb.search(query1 & query2))

    assert ids_or == ids_1 | ids_2
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
)  # fmt: skip
@pytest.mark.skipif(cannot_connect_to(RCSB_URL), reason="RCSB PDB is not available")
def test_search_return_type(return_type, expected):
    query = rcsb.BasicQuery(TC5B_TERM)
    assert rcsb.search(query, return_type) == expected
    assert rcsb.count(query, return_type) == len(expected)


@pytest.mark.skipif(cannot_connect_to(RCSB_URL), reason="RCSB PDB is not available")
@pytest.mark.parametrize("seed", np.arange(5))
def test_search_range(seed):
    query = rcsb.FieldQuery(
        "rcsb_entity_host_organism.scientific_name", exact_match="Homo sapiens"
    )
    count = rcsb.count(query)
    ref_entries = rcsb.search(query)
    assert len(ref_entries) == count

    np.random.seed(seed)
    range = sorted(np.random.choice(count, 2, replace=False))
    if range[1] - range[0] > 10000:
        # pagination only supports up to 10000 entries
        # (https://search.rcsb.org/#pagination)
        range[1] = range[0] + 10000
    test_entries = rcsb.search(query, range=range)

    assert test_entries == ref_entries[range[0] : range[1]]


@pytest.mark.skipif(cannot_connect_to(RCSB_URL), reason="RCSB PDB is not available")
@pytest.mark.parametrize("as_sorting_object", [False, True])
def test_search_sort(as_sorting_object):
    query = rcsb.FieldQuery(
        "rcsb_entity_host_organism.scientific_name", exact_match="Homo sapiens"
    )
    if as_sorting_object:
        sort_by = rcsb.Sorting("reflns.d_resolution_high", descending=False)
    else:
        sort_by = "reflns.d_resolution_high"
    entries = rcsb.search(query, sort_by=sort_by)

    resolutions = []
    for pdb_id in entries[:5]:
        pdbx_file = pdbx.BinaryCIFFile.read(rcsb.fetch(pdb_id, "bcif"))
        resolutions.append(pdbx_file.block["reflns"]["d_resolution_high"].as_item())

    if as_sorting_object:
        # In the tested case the Sorting object uses ascending order
        assert resolutions == list(sorted(resolutions))
    else:
        # Check if values are sorted in descending order
        assert resolutions == list(reversed(sorted(resolutions)))


@pytest.mark.skipif(cannot_connect_to(RCSB_URL), reason="RCSB PDB is not available")
def test_search_content_types():
    # Query to limit the number of returned results
    # for improved performance
    query = rcsb.FieldQuery(
        "rcsb_entity_host_organism.scientific_name", exact_match="Homo sapiens"
    )
    experimental_set = set(rcsb.search(query, content_types=["experimental"]))
    computational_set = set(rcsb.search(query, content_types=["computational"]))
    combined_set = set(
        rcsb.search(query, content_types=["experimental", "computational"])
    )

    # If there are no results, the following tests make no sense
    assert len(combined_set) > 0
    # There should be no common elements
    assert len(experimental_set & computational_set) == 0
    # The combined search should include the contents of both searches
    assert len(experimental_set | computational_set) == len(combined_set)

    assert rcsb.count(query, content_types=["experimental"]) == len(experimental_set)
    assert rcsb.count(query, content_types=["computational"]) == len(computational_set)
    assert rcsb.count(query, content_types=["experimental", "computational"]) == len(
        combined_set
    )

    # Expect an exception if no content_type
    with pytest.raises(ValueError):
        rcsb.search(query, content_types=[])
    with pytest.raises(ValueError):
        rcsb.count(query, content_types=[])


@pytest.mark.skipif(cannot_connect_to(RCSB_URL), reason="RCSB PDB is not available")
@pytest.mark.parametrize(
    "grouping, resolution_threshold, return_type, ref_groups",
    [
        (
            rcsb.IdentityGrouping(
                100, sort_by="rcsb_accession_info.initial_release_date"
            ),
            0.7,
            "polymer_entity",
            set(
                [
                    ("3X2M_1",),
                    ("6E6O_1",),
                    ("1YK4_1",),
                    ("5NW3_1",),
                    ("1US0_1",),
                    ("4HP2_1",),
                    ("2DSX_1",),
                    ("2VB1_1",),
                    ("7VOS_1", "3A38_1", "5D8V_1"),
                    ("1UCS_1",),
                    ("3NIR_1", "9EWK_1", "1EJG_1"),
                ]
            ),
        ),
        (
            rcsb.UniprotGrouping(sort_by="rcsb_accession_info.initial_release_date"),
            0.7,
            "polymer_entity",
            set(
                [
                    ("3X2M_1",),
                    ("6E6O_1",),
                    ("1YK4_1",),
                    ("5NW3_1",),
                    ("1US0_1",),
                    ("4HP2_1",),
                    ("2DSX_1",),
                    ("2VB1_1",),
                    ("7VOS_1", "3A38_1", "5D8V_1"),
                    ("1UCS_1",),
                    ("3NIR_1", "9EWK_1", "1EJG_1"),
                ]
            ),
        ),
        (
            rcsb.DepositGrouping(sort_by="rcsb_accession_info.initial_release_date"),
            0.9,
            "entry",
            set([("5R32",), ("5RDH", "5RBR"), ("7G0Z", "7FXV")]),
        ),
    ],
)
def test_search_grouping(grouping, resolution_threshold, return_type, ref_groups):
    """
    Check whether the same result as in a known example is achieved.
    """
    query = (
        rcsb.FieldQuery("exptl.method", exact_match="X-RAY DIFFRACTION")
        & rcsb.FieldQuery(
            "rcsb_entry_info.resolution_combined",
            range_closed=(0.0, resolution_threshold),
        )
        & rcsb.FieldQuery(
            "rcsb_accession_info.initial_release_date",
            less_or_equal=CUTOFF_DATE.isoformat(),
        )
    )

    test_groups = list(
        rcsb.search(query, return_type, group_by=grouping, return_groups=True).values()
    )
    test_representatives = rcsb.search(
        query, return_type, group_by=grouping, return_groups=False
    )
    test_count = rcsb.count(query, return_type, group_by=grouping)

    # List is not hashable
    assert set([tuple(group) for group in test_groups]) == ref_groups
    assert set(test_representatives) == set([group[0] for group in ref_groups])
    assert test_count == len(ref_groups)


@pytest.mark.skipif(cannot_connect_to(RCSB_URL), reason="RCSB PDB is not available")
def test_search_empty():
    query = rcsb.BasicQuery("This will not match any ID")
    assert rcsb.search(query) == []
    assert rcsb.count(query) == 0


@pytest.mark.parametrize(
    "field, params",
    [("invalid.field", {"exact_match": "Some Value"}), ("exptl.method", {"less": 5})],
)
@pytest.mark.skipif(cannot_connect_to(RCSB_URL), reason="RCSB PDB is not available")
def test_search_invalid(field, params):
    invalid_query = rcsb.FieldQuery(field, **params)
    with pytest.raises(RequestError, match="400"):
        rcsb.search(invalid_query)
    with pytest.raises(RequestError, match="400"):
        rcsb.count(invalid_query)
