# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
import itertools
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.io as strucio
from ..util import data_dir


# Result should be independent of cell size
@pytest.mark.parametrize("cell_size", [0.5, 1, 2, 5, 10])
def test_get_atoms(cell_size):
    """
    Test the correct functionality of a cell list on a simple test case
    with known solutions.
    """
    array = struc.AtomArray(length=5)
    array.coord = np.array([[0,0,i] for i in range(5)])
    cell_list = struc.CellList(array, cell_size=cell_size)
    assert cell_list.get_atoms(np.array([0,0,0.1]), 1).tolist() == [0,1]
    assert cell_list.get_atoms(np.array([0,0,1.1]), 1).tolist() == [1,2]
    assert cell_list.get_atoms(np.array([0,0,1.1]), 2).tolist() == [0,1,2,3]
    # Multiple positions
    pos = np.array([[0,0,0.1],
                    [0,0,1.1],
                    [0,0,4.1]])
    expected_indices = [0, 1, 2,
                        0, 1, 2, 3,
                        3, 4]
    indices = cell_list.get_atoms(pos, 2)
    assert indices[indices != -1].tolist() == expected_indices
    # Multiple positions and multiple radii
    pos = np.array([[0,0,0.1],
                    [0,0,1.1],
                    [0,0,4.1]])
    rad = np.array([1.0, 2.0, 3.0])
    expected_indices = [0, 1,
                        0, 1, 2, 3,
                        2, 3, 4]
    indices = cell_list.get_atoms(pos, rad)
    assert indices[indices != -1].tolist() == expected_indices


@pytest.mark.parametrize(
    "cell_size, threshold, periodic, use_selection",
    itertools.product(
        [0.5, 1, 2, 5, 10],
        [2, 5, 10],
        [False, True],
        [False, True],
    )
)
def test_adjacency_matrix(cell_size, threshold, periodic, use_selection):
    """
    Compare the construction of an adjacency matrix using a cell list
    and using a computationally expensive but simpler distance matrix.
    """
    array = strucio.load_structure(join(data_dir("structure"), "3o5r.mmtf"))
    
    if periodic:
        # Create an orthorhombic box
        # with the outer coordinates as bounds
        array.box = np.diag(
            np.max(array.coord, axis=-2) - np.min(array.coord, axis=-2)
        )

    if use_selection:
        np.random.seed(0)
        selection = np.random.choice((False, True), array.array_length())
    else:
        selection = None

    cell_list = struc.CellList(
        array, cell_size=cell_size, periodic=periodic, selection=selection
    )
    test_matrix = cell_list.create_adjacency_matrix(threshold)
    
    length = array.array_length()
    distance = struc.index_distance(
        array,
        np.stack(
            [
                np.repeat(np.arange(length), length),
                  np.tile(np.arange(length), length)
            ],
            axis=-1
        ),
        periodic
    )
    distance = np.reshape(distance, (length, length))
    # Create adjacency matrix from distance matrix
    exp_matrix = (distance <= threshold)
    if use_selection:
        # Set rows and columns to False for filtered out atoms
        exp_matrix[~selection, :] = False
        exp_matrix[:, ~selection] = False
    
    # Both ways to create an adjacency matrix
    # should give the same result
    assert np.array_equal(test_matrix, exp_matrix)


def test_outside_location():
    """
    Test result for location outside any cell.
    """
    array = strucio.load_structure(join(data_dir("structure"), "3o5r.mmtf"))
    array = array[struc.filter_amino_acids(array)]
    cell_list = struc.CellList(array, cell_size=5)
    outside_coord = np.min(array.coord, axis=0) - 100
    # Expect empty array
    assert len(cell_list.get_atoms(outside_coord, 5)) == 0


def test_selection():
    """
    Test whether the `selection` parameter in the constructor works.
    This is tested by comparing the selection done prior to cell list
    creation with the selection done in the cell list construction.
    """
    array = strucio.load_structure(join(data_dir("structure"), "3o5r.mmtf"))
    selection = np.array([False, True] * (array.array_length() // 2))
    
    # Selection prior to cell list creation
    selected = array[selection]
    cell_list = struc.CellList(selected, cell_size=10)
    ref_near_atoms = selected[cell_list.get_atoms(array.coord[0], 20.0)]

    # Selection in cell list creation
    cell_list = struc.CellList(array, cell_size=10, selection=selection)
    test_near_atoms = array[cell_list.get_atoms(array.coord[0], 20.0)]

    assert test_near_atoms == ref_near_atoms


def test_empty_coordinates():
    """
    Test whether empty input coordinates result in an empty output
    array/mask.
    """
    array = strucio.load_structure(join(data_dir("structure"), "3o5r.mmtf"))
    cell_list = struc.CellList(array, cell_size=10)
    
    for method in (
        struc.CellList.get_atoms, struc.CellList.get_atoms_in_cells
    ):
        indices = method(cell_list, np.array([]), 1, as_mask=False)
        mask = method(cell_list, np.array([]), 1, as_mask=True)
        assert len(indices) == 0
        assert len(mask) == 0
        assert indices.dtype == np.int32
        assert mask.dtype == bool