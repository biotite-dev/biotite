# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.io.mmtf as mmtf
from .util import data_dir


@pytest.fixture(
    params=[False, True] # as_negative
)
def bond_list(request):
    as_negative = request.param
    bond_array = np.array([(0,1),(2,1),(3,1),(3,4),(3,1),(1,2),(4,0),(6,4)])
    if as_negative:
        return struc.BondList(7, -7 + bond_array)
    else:
        return struc.BondList(7, bond_array)


def test_creation(bond_list):
    # Test includes redundancy removal and max bonds calculation
    assert bond_list.as_array().tolist() == [[0, 1, 0],
                                             [1, 2, 0],
                                             [1, 3, 0],
                                             [3, 4, 0],
                                             [0, 4, 0],
                                             [4, 6, 0]]
    assert bond_list._max_bonds_per_atom == 3
    assert bond_list._atom_count == 7


def test_modification(bond_list):
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


def test_access(bond_list):
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
    merged_list = bond_list.merge(struc.BondList(8, np.array([(4,6),(6,7)])))
    assert merged_list.as_array().tolist() == [[0, 1, 0],
                                               [1, 2, 0],
                                               [1, 3, 0],
                                               [3, 4, 0],
                                               [0, 4, 0],
                                               [4, 6, 0],
                                               [6, 7, 0]]


def test_concatenation(bond_list):
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


def test_atom_array_consistency():
    array = strucio.load_structure(join(data_dir, "1l2y.mmtf"))[0]
    ca = array[array.atom_name == "CA"]
    # Just for testing, does not refelct real bonds
    bond_list = struc.BondList(ca.array_length(), 
        np.array([(0,1),(2,8),(5,15),(1,5),(0,9),(3,18),(2,9)])
    )
    # The bonds, should always point to the same atoms (same res_id),
    # irrespective of indexing
    ids1 = ca.res_id[bond_list.as_array()[:,:2].flatten()]
    # Some random boolean mask as index
    mask = np.array([1,1,1,1,0,1,0,0,1,1,0,1,1,0,0,1,1,0,1,1], dtype=np.bool)
    ca = ca[mask]
    bond_list = bond_list[mask]
    ids2 = ca.res_id[bond_list.as_array()[:,:2].flatten()]
    assert ids1.tolist() == ids2.tolist()


@pytest.mark.parametrize("single_model", [False, True])
def test_connect_via_residue_names(single_model):
    """
    Test whether the created bond list is equal to the bonds deposited
    in the MMTF file.
    """
    # Structure with peptide, nucleotide, small molecules and water
    file = mmtf.MMTFFile()
    file.read(join(data_dir, "5ugo.mmtf"))
    if single_model:
        atoms = mmtf.get_structure(file, include_bonds=True, model=1)
    else:
        atoms = mmtf.get_structure(file, include_bonds=True)
    
    ref_bonds = atoms.bonds

    test_bonds = struc.connect_via_residue_names(atoms)

    assert test_bonds == ref_bonds


def test_connect_via_distances():
    """
    Test whether the created bond list is equal to the bonds deposited
    in the MMTF file.
    """
    file = mmtf.MMTFFile()
    file.read(join(data_dir, "1l2y.mmtf"))
    atoms = mmtf.get_structure(file, include_bonds=True, model=1)
    # Remove termini to solve the issue that the reference bonds do not
    # contain proper bonds for the protonated/deprotonated termini
    atoms = atoms[(atoms.res_id > 1) & (atoms.res_id < 20)]
    
    ref_bonds = atoms.bonds
    # Convert all bonds to BondType.ANY
    ref_bonds = struc.BondList(
        ref_bonds.get_atom_count(), ref_bonds.as_array()[:, :2]
    )

    test_bonds = struc.connect_via_distances(atoms)

    assert test_bonds == ref_bonds