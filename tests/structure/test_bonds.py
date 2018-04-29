# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np
from os.path import join
from .util import data_dir
import pytest

@pytest.fixture
def bond_list():
    bond_array = np.array([(0,1),(2,1),(3,1),(3,4),(3,1),(1,2),(4,0),(6,4)])
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
    array = strucio.get_structure_from(join(data_dir, "1l2y.mmtf"))[0]
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