# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import pytest
import numpy as np
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.io.pdb as pdb
import biotite.database.rcsb as rcsb
from tempfile import gettempdir
from biotite.structure.basepairs import base_pairs, base_pairs_edge, map_nucleotide, edge
from biotite.structure.info import residue
from os.path import join
from ..util import data_dir
import json


def reversed_iterator(iter):
    """
    Returns a reversed list of the elements of an Iterator.
    """
    return reversed(list(iter))


@pytest.fixture
def nuc_sample_array():
    return strucio.load_structure(join(data_dir("structure"), "1qxb.cif"))

@pytest.fixture
def basepairs(nuc_sample_array):
    """
    Generate a test output for the base_pairs function.
    """
    residue_indices, residue_names = struc.residues.get_residues(
        nuc_sample_array
    )[0:24]
    return np.vstack((residue_indices[:12], np.flip(residue_indices)[:12])).T


def check_output(computed_basepairs, basepairs):
    """
    Check the output of base_pairs.
    """

    # Check if basepairs are unique in computed_basepairs
    seen = set()
    assert (not any(
        (base1, base2) in seen) or (base2, base1 in seen)
        or seen.add((base1, base2)) for base1, base2 in computed_basepairs
        )
    # Check if the right number of basepairs is in computed_basepairs
    assert(len(computed_basepairs) == len(basepairs))
    # Check if the right basepairs are in computed_basepairs
    for comp_basepair in computed_basepairs:
        assert ((comp_basepair in basepairs) \
                or (comp_basepair in np.flip(basepairs)))

@pytest.mark.parametrize("unique_bool", [False, True])
def test_base_pairs_forward(nuc_sample_array, basepairs, unique_bool):
    """
    Test for the function base_pairs.
    """
    computed_basepairs = base_pairs(nuc_sample_array, unique=unique_bool)
    check_output(nuc_sample_array[computed_basepairs].res_id, basepairs)


def test_base_pairs_forward_no_hydrogen(nuc_sample_array, basepairs):
    """
    Test for the function base_pairs with the hydrogens removed from the
    test structure.
    """
    nuc_sample_array = nuc_sample_array[nuc_sample_array.element != "H"]
    computed_basepairs = base_pairs(nuc_sample_array)
    check_output(nuc_sample_array[computed_basepairs].res_id, basepairs)

@pytest.mark.parametrize("unique_bool", [False, True])
def test_base_pairs_reverse(nuc_sample_array, basepairs, unique_bool):
    """
    Reverse the order of residues in the atom_array and then test the
    function base_pairs.
    """

    # Reverse sequence of residues in nuc_sample_array
    reversed_nuc_sample_array = struc.AtomArray(0)
    for residue in reversed_iterator(struc.residue_iter(nuc_sample_array)):
        reversed_nuc_sample_array = reversed_nuc_sample_array + residue

    computed_basepairs = base_pairs(
        reversed_nuc_sample_array, unique=unique_bool
    )
    check_output(
        reversed_nuc_sample_array[computed_basepairs].res_id, basepairs
    )

def test_base_pairs_reverse_no_hydrogen(nuc_sample_array, basepairs):
    """
    Remove the hydrogens from the sample structure. Then reverse the
    order of residues in the atom_array and then test the function
    base_pairs.
    """
    nuc_sample_array = nuc_sample_array[nuc_sample_array.element != "H"]
    # Reverse sequence of residues in nuc_sample_array
    reversed_nuc_sample_array = struc.AtomArray(0)
    for residue in reversed_iterator(struc.residue_iter(nuc_sample_array)):
        reversed_nuc_sample_array = reversed_nuc_sample_array + residue

    computed_basepairs = base_pairs(reversed_nuc_sample_array)
    check_output(
        reversed_nuc_sample_array[computed_basepairs].res_id, basepairs
    )

@pytest.mark.parametrize("seed", range(10))
def test_base_pairs_reordered(nuc_sample_array, seed):
    """
    Test the function base_pairs with structure where the atoms are not
    in the RCSB-Order.
    """
    # Randomly reorder the atoms in each residue
    nuc_sample_array_reordered = struc.AtomArray(0)
    np.random.seed(seed)

    for residue in struc.residue_iter(nuc_sample_array):
        bound = residue.array_length()
        indices = np.random.choice(
            np.arange(bound), bound,replace=False
        )
        nuc_sample_array_reordered += residue[..., indices]

    assert(np.all(
        struc.base_pairs(nuc_sample_array)
        == struc.base_pairs(nuc_sample_array_reordered)
    ))

def test_map_nucleotide():
    """Test the function map_nucleotide with some examples.
    """
    pyrimidines = ['C', 'T', 'U']
    purines = ['A', 'G']

    # Test that the standard bases are correctly identified
    assert map_nucleotide(residue('U')) == ('U', True)
    assert map_nucleotide(residue('A')) == ('A', True)
    assert map_nucleotide(residue('T')) == ('T', True)
    assert map_nucleotide(residue('G')) == ('G', True)
    assert map_nucleotide(residue('C')) == ('C', True)

    # Test that some non_standard nucleotides are mapped correctly to
    # pyrimidine/purine references
    psu_tuple = map_nucleotide(residue('PSU'))
    assert psu_tuple[0] in pyrimidines
    assert psu_tuple[1] == False

    psu_tuple = map_nucleotide(residue('3MC'))
    assert psu_tuple[0] in pyrimidines
    assert psu_tuple[1] == False

    i_tuple = map_nucleotide(residue('I'))
    assert i_tuple[0] in purines
    assert i_tuple[1] == False

    m7g_tuple = map_nucleotide(residue('M7G'))
    assert m7g_tuple[0] in purines
    assert m7g_tuple[1] == False

    assert map_nucleotide(residue('ALA')) is None

@pytest.fixture
def reference_edges():
    """Gets the reference edges from specified pdb files
    """
    reference = strucio.load_structure(
        join(data_dir("structure"), "base_pairs/1gid.cif")
    )

    with open(
        join(data_dir("structure"), "base_pairs/1gid.json"
    ), "r") as file:
        edges = np.array(json.load(file))
    #edges = json.loads(join(data_dir("structure"), "base_pairs/1gid.json"))
    return reference, edges

def test_base_pairs_edge(reference_edges):
    reference_structure, reference_edges = reference_edges
    pairs = base_pairs(reference_structure, unique=False)
    for pair_res_ids in reference_structure[pairs].res_id:
        print(pair_res_ids[0])
        print(pair_res_ids[1])
        if (
            np.any(
                np.logical_and(
                    reference_edges[:, 0] == pair_res_ids[0],
                    reference_edges[:, 1] == pair_res_ids[1]
                )

            ) or
            np.any(
                np.logical_and(
                    reference_edges[:, 0] == pair_res_ids[0],
                    reference_edges[:, 1] == pair_res_ids[1]
                )

            )
        ):
            print('found')
        else:
            print("not_found")
    """


    edges = base_pairs_edge(reference_structure, pairs)
    #print(reference_structure[pairs.flatten()].res_id)
    for pair in pairs:
        if np.any(reference_structure[pair].res_id == 108):
            print(reference_structure[pair[0]].res_id)
            print(reference_structure[pair[1]].res_id)

    for res, sedge in zip(reference_structure[pairs.flatten()].res_id, edges.flatten()):
        if sedge == edge.invalid:
            continue
        if str(res) in reference_edges:
            if edge(reference_edges[str(res)]) != sedge:
                print(res)
                print(edge(reference_edges[str(res)]))
                print(sedge)
        else:
            print(res)
            print(f"weird {sedge}")
    """