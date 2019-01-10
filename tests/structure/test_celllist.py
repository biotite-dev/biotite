# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from .util import data_dir
import biotite.structure as struc
import biotite.structure.io as strucio
from os.path import join
import itertools
import numpy as np
import pytest


# Result should be independent of cell size
@pytest.mark.parametrize("cell_size", [0.5, 1, 2, 5, 10])
def test_get_atoms(cell_size):
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
    "cell_size, threshold, periodic",
    itertools.product(
        [0.5, 1, 2, 5, 10],
        [2, 5, 10],
        [False, True]
    )
)
def test_adjacency_matrix(cell_size, threshold, periodic):
    """
    Compare the construction of an adjacency matrix using a cell list
    and using a computationally expensive but simpler distance matrix
    """
    array = strucio.load_structure(join(data_dir, "3o5r.mmtf"))
    if periodic:
        # Create an orthorhombic box
        # with the outer coordinates as bounds
        array.box = np.diag(
            np.max(array.coord, axis=-2) - np.min(array.coord, axis=-2)
        )
    cell_list = struc.CellList(array, cell_size=cell_size, periodic=periodic)
    matrix = cell_list.create_adjacency_matrix(threshold)
    
    # Create distance matrix
    # Convert to float64 to avoid errorenous warning
    # https://github.com/ContinuumIO/anaconda-issues/issues/9129
    array.coord = array.coord.astype(np.float64)
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
    expected_matrix = (distance <= threshold)
    
    # Both ways to create an adjacency matrix
    # should give the same result
    assert np.array_equal(matrix, expected_matrix)


def test_outside_location():
    # Test result for location outside any cell
    array = strucio.load_structure(join(data_dir, "3o5r.mmtf"))
    array = array[struc.filter_amino_acids(array)]
    cell_list = struc.CellList(array, cell_size=5)
    outside_coord = np.min(array.coord, axis=0) - 100
    # Expect empty array
    assert len(cell_list.get_atoms(outside_coord, 5)) == 0