# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.info as info
import biotite.structure.io as strucio
import biotite.structure.io.mmtf as mmtf
from ..util import data_dir


def generate_random_bond_list(atom_count, bond_count, seed=0):
    """
    Generate a random :class:`BondList`.
    """
    np.random.seed(seed)
    # Create random bonds between atoms of
    # a potential atom array of length ATOM_COUNT
    bonds = np.random.randint(atom_count, size=(bond_count, 3))
    # Clip bond types to allowed BondType values
    bonds[:, 2] %= len(struc.BondType)
    # Remove bonds of atoms to itself
    bonds = bonds[bonds[:,0] != bonds[:,1]]
    assert len(bonds) > 0
    return struc.BondList(atom_count, bonds)


@pytest.fixture(
    params=[False, True] # as_negative
)
def bond_list(request):
    """
    A toy :class:`BondList`.
    """
    as_negative = request.param
    bond_array = np.array([(0,1),(2,1),(3,1),(3,4),(3,1),(1,2),(4,0),(6,4)])
    if as_negative:
        return struc.BondList(7, -7 + bond_array)
    else:
        return struc.BondList(7, bond_array)


def test_creation(bond_list):
    """
    Test creating a :class:`BondList` on a known example.
    """
    # Test includes redundancy removal and max bonds calculation
    assert bond_list.as_array().tolist() == [[0, 1, 0],
                                             [1, 2, 0],
                                             [1, 3, 0],
                                             [3, 4, 0],
                                             [0, 4, 0],
                                             [4, 6, 0]]
    assert bond_list._max_bonds_per_atom == 3
    assert bond_list._atom_count == 7


def test_invalid_creation():
    """
    Check for appropriate errors when invalid input is given to the
    :class:`BondLst` constructor.
    """
    # Test invalid input shapes
    with pytest.raises(ValueError):
        struc.BondList(
            5,
            np.array([
                [1,2,3,4]
            ])
        )
    with pytest.raises(ValueError):
        struc.BondList(
            5,
            np.array([1,2])
        )

    # Test invalid atom indices
    with pytest.raises(IndexError):
        struc.BondList(
            5,
            np.array([
                [1,2],
                # 5 is an invalid index for an atom count of 5
                [5,2]
            ])
        )
    with pytest.raises(IndexError):
        struc.BondList(
            5,
            np.array([
                # Index -6 is invalid for an atom count of 5
                [-6,3],
                [3,4]
            ])
        )
    
    # Test invalid BondType
    with pytest.raises(ValueError):
        struc.BondList(
            5,
            np.array([
                # BondType '8' does not exist
                [1,2,8]
            ])
        )


def test_modification(bond_list):
    """
    Test whether `BondList` correctly identifies whether it contains a
    certain bond on a known example.
    """
    # Already in list
    bond_list.add_bond(3, 1)
    # Also already in list -> update
    bond_list.add_bond(1, 3, 1)
    # The same but with negative atom index
    bond_list.add_bond(-6, -4, 1)
    # Not in list
    bond_list.add_bond(4, 1)
    # In list -> remove
    bond_list.remove_bond(4, 0)
    # Not in list -> Do nothing
    bond_list.remove_bond(0, 3)
    # Remove mutliple bonds, one of them is not in list
    bond_list.remove_bonds(struc.BondList(10, np.array([(1,0),(1,2),(8,9)])))
    assert bond_list.as_array().tolist() == [[1, 3, 1],
                                             [3, 4, 0],
                                             [4, 6, 0],
                                             [1, 4, 0]]


def test_contains(bond_list):
    """
    Test whether `BondList` correctly identifies whether it contains a
    certain bond on a known example.
    """
    for atom1, atom2, _ in bond_list.as_array():
        assert (atom1, atom2) in bond_list
        assert (atom2, atom1) in bond_list
    # Negative test
    assert (0, 5) not in bond_list


def test_access(bond_list):
    """
    Test getting bonds from a `BondList` objects on a known example.
    """
    # Bigger challenge with different bond types
    bond_list.add_bond(1, 3, 1)
    bonds, bond_types = bond_list.get_bonds(0)
    assert bonds.tolist() == [1, 4]
    assert bond_types.tolist() == [0, 0]
    bonds, bond_types = bond_list.get_bonds(1)
    assert bonds.tolist() == [0, 2, 3]
    assert bond_types.tolist() == [0, 0, 1]
    bonds, bond_types = bond_list.get_bonds(2)
    assert bonds.tolist() == [1]
    assert bond_types.tolist() == [0]
    bonds, bond_types = bond_list.get_bonds(3)
    assert bonds.tolist() == [1, 4]
    assert bond_types.tolist() == [1, 0]
    bonds, bond_types = bond_list.get_bonds(4)
    assert bonds.tolist() == [3, 0, 6]
    assert bond_types.tolist() == [0, 0, 0]


def test_merge(bond_list):
    """
    Test merging two `BondList` objects on a known example.
    """
    merged_list = bond_list.merge(struc.BondList(8, np.array([(4,6),(6,7)])))
    assert merged_list.as_array().tolist() == [[0, 1, 0],
                                               [1, 2, 0],
                                               [1, 3, 0],
                                               [3, 4, 0],
                                               [0, 4, 0],
                                               [4, 6, 0],
                                               [6, 7, 0]]


def test_concatenation(bond_list):
    """
    Test concatenation of two `BondList` objects on a known example.
    """
    bond_list += struc.BondList(3, np.array([(0,1,2),(1,2,2)]))
    assert bond_list.as_array().tolist() == [[0, 1, 0],
                                             [1, 2, 0],
                                             [1, 3, 0],
                                             [3, 4, 0],
                                             [0, 4, 0],
                                             [4, 6, 0],
                                             [7, 8, 2],
                                             [8, 9, 2]]
    assert bond_list._max_bonds_per_atom == 3
    assert bond_list._atom_count == 10


def test_indexing(bond_list):
    """
    Test indexing with different index types on a known example.
    """
    sub_list = bond_list[:]
    assert sub_list.as_array().tolist() == bond_list.as_array().tolist()
    sub_list = bond_list[1:6:2]
    assert sub_list.as_array().tolist() == [[0, 1, 0]]
    sub_list = bond_list[:4]
    assert sub_list.as_array().tolist() == [[0, 1, 0],
                                            [1, 2, 0],
                                            [1, 3, 0]]
    sub_list = bond_list[2:]
    assert sub_list.as_array().tolist() == [[1, 2, 0],
                                            [2, 4, 0]]
    
    sub_list = bond_list[[0,3,4]]
    assert sub_list.as_array().tolist() == [[1, 2, 0],
                                            [0, 2, 0]]

    sub_list = bond_list[np.array([True,False,False,True,True,False,True])]
    assert sub_list.as_array().tolist() == [[1, 2, 0],
                                            [0, 2, 0],
                                            [2, 3, 0]]


def test_get_all_bonds():
    """
    Test whether the individual rows returned from
    :func:`get_all_bonds()` are equal to corresponding calls of
    :func:`get_bonds()`.
    """
    ATOM_COUNT = 100
    BOND_COUNT = 500
    
    bond_list = generate_random_bond_list(ATOM_COUNT, BOND_COUNT)

    bonds, bond_types = bond_list.get_all_bonds()

    assert (bonds != -1).all(axis=1).any(axis=0)
    assert (bond_types != -1).all(axis=1).any(axis=0)

    test_bonds = [
        (
            bonded_i[bonded_i != -1].tolist(),
            bond_type[bond_type != -1].tolist()
        )
        for bonded_i, bond_type in zip(bonds, bond_types)
    ]

    ref_bonds = [bond_list.get_bonds(i) for i in range(ATOM_COUNT)]
    ref_bonds = [
        (bonded_i.tolist(), bond_type.tolist())
        for bonded_i, bond_type in ref_bonds
    ]

    assert test_bonds == ref_bonds


def test_adjacency_matrix():
    """
    Test whether the matrix created from :func:`adjacency_matrix()`
    is equal to a manually created matrix using :func:`get_bonds()`.
    """
    ATOM_COUNT = 100
    BOND_COUNT = 500
    
    bond_list = generate_random_bond_list(ATOM_COUNT, BOND_COUNT)

    test_matrix = bond_list.adjacency_matrix()

    ref_matrix = np.zeros((ATOM_COUNT, ATOM_COUNT), dtype=bool)
    for i in range(ATOM_COUNT):
        for j, _ in zip(*bond_list.get_bonds(i)):
            ref_matrix[i, j] = True
    
    assert test_matrix.tolist() == ref_matrix.tolist()


def test_bond_type_matrix():
    """
    Test whether the matrix created from :func:`bond_type_matrix()`
    is equal to a manually created matrix using :func:`get_bonds()`.
    """
    ATOM_COUNT = 100
    BOND_COUNT = 500
    
    bond_list = generate_random_bond_list(ATOM_COUNT, BOND_COUNT)

    test_matrix = bond_list.bond_type_matrix()

    ref_matrix = np.full((ATOM_COUNT, ATOM_COUNT), -1, dtype=int)
    for i in range(ATOM_COUNT):
        for j, bond_type in zip(*bond_list.get_bonds(i)):
            ref_matrix[i, j] = bond_type
    
    assert test_matrix.tolist() == ref_matrix.tolist()


def test_sorted_array_indexing():
    """
    Test whether indexing with an sorted index array results in the
    same `BondList` as indexing with an equivalent boolean mask.
    """
    ATOM_COUNT = 100
    BOND_COUNT = 500
    INDEX_SIZE = 80

    bonds = generate_random_bond_list(ATOM_COUNT, BOND_COUNT)

    # Create a sorted array of random indices for the BondList
    # Indices may not occur multiple times -> 'replace=False'
    index_array = np.sort(np.random.choice(
        np.arange(ATOM_COUNT), INDEX_SIZE, replace=False
    ))
    test_bonds = bonds[index_array]

    # Create a boolean mask that indexes the same elements as the array 
    mask = np.zeros(ATOM_COUNT, dtype=bool)
    mask[index_array] = True
    ref_bonds = bonds[mask]

    assert test_bonds.as_array().tolist() == ref_bonds.as_array().tolist()


def test_unsorted_array_indexing():
    """
    Test whether indexing with an unsorted index array results in the
    same bonded atoms, as indexing with a sorted index array.
    The `BondList` objects should be different because the actual atom
    order is different, but the pointed atoms should still be the same,
    if the respective index is also applied to the atoms.
    """
    ATOM_COUNT = 100
    BOND_COUNT = 500
    INDEX_SIZE = 80

    bonds = generate_random_bond_list(ATOM_COUNT, BOND_COUNT)
    # For simplicity use a reference integer array
    # instead of an atom array
    integers = np.arange(ATOM_COUNT)
    
    # Create random bonds between the reference integers
    bonds = np.random.randint(ATOM_COUNT, size=(BOND_COUNT, 2))
    # Remove bonds of elements to itself
    bonds = bonds[bonds[:,0] != bonds[:,1]]
    assert len(bonds) > 0
    bonds = struc.BondList(ATOM_COUNT, bonds)

    # Create an unsorted array of random indices for the BondList
    # Indices should be unsorted -> 'replace=False'
    unsorted_index = np.random.choice(
        np.arange(ATOM_COUNT), INDEX_SIZE, replace=False
    )
    test_bonds = bonds[unsorted_index]

    # Create a sorted variant of the index array
    sorted_index = np.sort(unsorted_index)
    # Check whether the unsorted array is really unsorted
    assert sorted_index.tolist() != unsorted_index.tolist()
    ref_bonds = bonds[sorted_index]

    unsorted_indexed_integers = integers[unsorted_index]
    sorted_indexed_integers = integers[sorted_index]
    # Get the 'atoms', in this case integers, that are connected with a bond
    # Use a set for simpler comparison between the sorted and unsorted variant
    # Omit the bond type -> 'bonds.as_array()[:, :2]'
    test_integer_pairs = set([
        frozenset((unsorted_indexed_integers[i], unsorted_indexed_integers[j]))
        for i, j in test_bonds.as_array()[:, :2]
    ])
    ref_integer_pairs = set([
        frozenset((sorted_indexed_integers[i], sorted_indexed_integers[j]))
        for i, j in ref_bonds.as_array()[:, :2]
    ])

    # The BondList entries should be different,
    # since they point to different positions in the reference array
    assert test_bonds.as_array().tolist() != ref_bonds.as_array().tolist()
    # But the actual bonded 'atom' pairs, should still be the same
    assert test_integer_pairs == ref_integer_pairs
    # Additionally, check whether in each bond the lower atom index 
    # comes first
    for i, j, _ in test_bonds.as_array():
        assert i < j


def test_atom_array_consistency():
    """
    Test whether the associated bonds of an `AtomArray` still point to
    the same atoms after indexing with a boolean mask.
    The boolean mask is constructed in a way that all bonded atoms are
    masked.
    """
    array = strucio.load_structure(join(data_dir("structure"), "1l2y.mmtf"))[0]
    ca = array[array.atom_name == "CA"]
    # Just for testing, does not reflect real bonds
    bond_list = struc.BondList(ca.array_length(), 
        np.array([(0,1),(2,8),(5,15),(1,5),(0,9),(3,18),(2,9)])
    )
    ca.bonds = bond_list
    
    ref_ids = ca.res_id[bond_list.as_array()[:,:2].flatten()]
    
    # Some random boolean mask as index,
    # but all bonded atoms are included
    mask = np.array([1,1,1,1,0,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1], dtype=bool)
    masked_ca = ca[mask]
    test_ids = masked_ca.res_id[masked_ca.bonds.as_array()[:,:2].flatten()]
    
    # The bonds, should always point to the same atoms (same res_id),
    # irrespective of indexing
    assert test_ids.tolist() == ref_ids.tolist()


@pytest.mark.parametrize("single_model", [False, True])
def test_connect_via_residue_names(single_model):
    """
    Test whether the created bond list is equal to the bonds deposited
    in the MMTF file.
    """
    # Structure with peptide, nucleotide, small molecules and water
    file = mmtf.MMTFFile.read(join(data_dir("structure"), "5ugo.mmtf"))
    if single_model:
        atoms = mmtf.get_structure(file, include_bonds=True, model=1)
    else:
        atoms = mmtf.get_structure(file, include_bonds=True)
    
    ref_bonds = atoms.bonds

    test_bonds = struc.connect_via_residue_names(atoms)
    # MMTF format does not represent aromaticity
    test_bonds.remove_aromaticity()

    assert test_bonds == ref_bonds


@pytest.mark.parametrize("periodic", [False, True])
def test_connect_via_distances(periodic):
    """
    Test whether the created bond list is equal to the bonds deposited
    in the MMTF file.
    """
    file = mmtf.MMTFFile.read(join(data_dir("structure"), "1l2y.mmtf"))
    atoms = mmtf.get_structure(file, include_bonds=True, model=1)
    # Remove termini to solve the issue that the reference bonds do not
    # contain proper bonds for the protonated/deprotonated termini
    atoms = atoms[(atoms.res_id > 1) & (atoms.res_id < 20)]

    if periodic:
        # Add large dummy box to test parameter
        # No actual bonds over the periodic boundary are expected
        atoms.box = np.identity(3) * 100
    
    ref_bonds = atoms.bonds
    # Convert all bonds to BondType.ANY
    ref_bonds = struc.BondList(
        ref_bonds.get_atom_count(), ref_bonds.as_array()[:, :2]
    )

    test_bonds = struc.connect_via_distances(atoms, periodic=periodic)

    assert test_bonds == ref_bonds


def test_find_connected(bond_list):
    """
    Find all connected atoms to an atom in a known example.
    """
    for index in (0,1,2,3,4,6):
        assert struc.find_connected(bond_list, index).tolist() == [0,1,2,3,4,6]
    assert struc.find_connected(bond_list, 5).tolist() == [5]


@pytest.mark.parametrize(
    "res_name, expected_bonds",
    [
        # Easy ligand visualization at:
        # https://www.rcsb.org/ligand/<ABC>
        ("TYR", [
            ("N",   "CA" ),
            ("CA",  "C"  ),
            ("CA",  "CB" ),
            ("C",   "OXT"),
            ("CB",  "CG" ),
            ("CZ",  "OH" ),
        ]),
        ("CEL", [
            ("C1",   "C4" ),
            ("C8",   "C11"),
            ("C15",  "S1" ),
            ("N3",   "S1" )
        ]),
        ("LEO", [
            ("C3",   "C8" ),
            ("C6",   "C17"),
            ("C17",  "C22"),
        ]),
    ]
)
def test_find_rotatable_bonds(res_name, expected_bonds):
    """
    Check the :func:`find_rotatable_bonds()` function based on
    known examples.
    """
    molecule = info.residue(res_name)
    
    ref_bond_set = {
        tuple(sorted((name_i, name_j))) for name_i, name_j in expected_bonds
    }

    rotatable_bonds = struc.find_rotatable_bonds(molecule.bonds)
    test_bond_set = set()
    for i, j, _ in rotatable_bonds.as_array():
        test_bond_set.add(
            tuple(sorted((molecule.atom_name[i], molecule.atom_name[j])))
        )
    
    # Compare with reference bonded atom names
    assert test_bond_set == ref_bond_set
    # All rotatable bonds must be single bonds
    assert np.all(rotatable_bonds.as_array()[:, 2] == struc.BondType.SINGLE)