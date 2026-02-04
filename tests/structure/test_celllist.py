# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
from os.path import join
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.io as strucio
from tests.util import data_dir
import pickle


# Result should be independent of cell size
@pytest.mark.parametrize("cell_size", [0.5, 1, 2, 5, 10])
def test_get_atoms(cell_size):
    """
    Test the correct functionality of a cell list on a simple test case
    with known solutions.
    """
    array = struc.AtomArray(length=5)
    array.coord = np.array([[0, 0, i] for i in range(5)])
    cell_list = struc.CellList(array, cell_size=cell_size)
    assert cell_list.get_atoms(np.array([0, 0, 0.1]), 1).tolist() == [0, 1]
    assert cell_list.get_atoms(np.array([0, 0, 1.1]), 1).tolist() == [1, 2]
    assert cell_list.get_atoms(np.array([0, 0, 1.1]), 2).tolist() == [0, 1, 2, 3]
    # Multiple positions
    pos = np.array([[0, 0, 0.1], [0, 0, 1.1], [0, 0, 4.1]])
    expected_indices = [0, 1, 2, 0, 1, 2, 3, 3, 4]
    indices = cell_list.get_atoms(pos, 2)
    assert indices[indices != -1].tolist() == expected_indices
    # Multiple positions and multiple radii
    pos = np.array([[0, 0, 0.1], [0, 0, 1.1], [0, 0, 4.1]])
    rad = np.array([1.0, 2.0, 3.0])
    expected_indices = [0, 1, 0, 1, 2, 3, 2, 3, 4]
    indices = cell_list.get_atoms(pos, rad)
    assert indices[indices != -1].tolist() == expected_indices


@pytest.mark.parametrize(
    "cell_size, threshold, periodic, use_selection",
    itertools.product(
        [0.5, 1, 2, 5, 10],
        [2, 5, 10],
        [False, True],
        [False, True],
    ),
)
def test_adjacency_matrix(cell_size, threshold, periodic, use_selection):
    """
    Compare the construction of an adjacency matrix using a cell list
    and using a computationally expensive but simpler distance matrix.
    """
    array = strucio.load_structure(join(data_dir("structure"), "3o5r.bcif"))

    if periodic:
        # Create an orthorhombic box
        # with the outer coordinates as bounds
        array.box = np.diag(np.max(array.coord, axis=-2) - np.min(array.coord, axis=-2))

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
            [np.repeat(np.arange(length), length), np.tile(np.arange(length), length)],
            axis=-1,
        ),
        periodic,
    )
    distance = np.reshape(distance, (length, length))
    # Create adjacency matrix from distance matrix
    exp_matrix = distance <= threshold
    if use_selection:
        # Set rows and columns to False for filtered out atoms
        exp_matrix[~selection, :] = False
        exp_matrix[:, ~selection] = False

    # Both ways to create an adjacency matrix
    # should give the same result
    assert np.array_equal(test_matrix, exp_matrix)


@pytest.mark.parametrize("result_format", [struc.CellListResult.MAPPING, struc.CellListResult.MATRIX, struc.CellListResult.PAIRS])
def test_result_format_consistency(result_format):
    """
    Test that independent of the result format, the matrix constructed from it is
    consistent.
    """
    CELL_SIZE = 5

    atoms = strucio.load_structure(join(data_dir("structure"), "3o5r.bcif"))
    cell_list = struc.CellList(atoms, cell_size=CELL_SIZE)
    ref_matrix = cell_list.create_adjacency_matrix(threshold=CELL_SIZE)

    result = cell_list.get_atoms(atoms.coord, CELL_SIZE, result_format=result_format)
    match result_format:
        case struc.CellListResult.MAPPING:
            mapping = result
            test_matrix = np.zeros((atoms.array_length(), atoms.array_length()), dtype=bool)
            for i, adjacent_atoms in enumerate(mapping):
                for j in adjacent_atoms[adjacent_atoms != -1]:
                    test_matrix[i, j] = True
        case struc.CellListResult.MATRIX:
            test_matrix = result
        case struc.CellListResult.PAIRS:
            pairs = result
            test_matrix = np.zeros((atoms.array_length(), atoms.array_length()), dtype=bool)
            test_matrix[pairs[:, 0], pairs[:, 1]] = True

    assert np.array_equal(test_matrix, ref_matrix)


@pytest.mark.parametrize("displacement", [-1000, 1000])
def test_outside_location(displacement):
    """
    Test result for location outside any cell.
    """
    CELL_SIZE = 5

    array = strucio.load_structure(join(data_dir("structure"), "3o5r.bcif"))
    array = array[struc.filter_amino_acids(array)]
    cell_list = struc.CellList(array, cell_size=CELL_SIZE)
    outside_coord = np.mean(array.coord, axis=0) + displacement
    # Expect empty array
    assert len(cell_list.get_atoms(outside_coord, CELL_SIZE)) == 0


@pytest.mark.parametrize("radius", [2, 5, 10])
@pytest.mark.parametrize("method", [struc.CellList.get_atoms, struc.CellList.get_atoms_in_cells])
def test_single_and_multiple_radii(radius, method):
    """
    Check if getting neighbors with multiple radii results in the same result as getting neighbors
    with a single radius for each coordinate.
    """
    CELL_SIZE = 5
    RADIUS_RANGE = (2, 10)
    N_SAMPLES = 100

    atoms = strucio.load_structure(join(data_dir("structure"), "3o5r.bcif"))
    cell_list = struc.CellList(atoms, cell_size=CELL_SIZE)

    # Pick random radii
    rng = np.random.default_rng(0)
    radii = rng.integers(RADIUS_RANGE[0], RADIUS_RANGE[1], size=N_SAMPLES)

    mutliple_radius_result = method(cell_list, atoms.coord[:N_SAMPLES], radii, result_format=struc.CellListResult.MAPPING)
    max_neighbors = mutliple_radius_result.shape[-1]

    multiple_radius_result = method(cell_list, atoms.coord, radii, result_format=struc.CellListResult.MAPPING)

    single_radius_result = np.zeros((N_SAMPLES, max_neighbors), dtype=int)
    for i in range(N_SAMPLES):
        result = method(cell_list, atoms.coord[i], radii[i], result_format=struc.CellListResult.MAPPING)
        single_radius_result[i, :len(result)] = result

    assert np.array_equal(multiple_radius_result, single_radius_result)


def test_selection():
    """
    Test whether the `selection` parameter in the constructor works.
    This is tested by comparing the selection done prior to cell list
    creation with the selection done in the cell list construction.
    """
    array = strucio.load_structure(join(data_dir("structure"), "3o5r.bcif"))
    selection = np.array([False, True] * (array.array_length() // 2))

    # Selection prior to cell list creation
    selected = array[selection]
    cell_list = struc.CellList(selected, cell_size=10)
    ref_near_atoms = selected[cell_list.get_atoms(array.coord[0], 20.0)]

    # Selection in cell list creation
    cell_list = struc.CellList(array, cell_size=10, selection=selection)
    test_near_atoms = array[cell_list.get_atoms(array.coord[0], 20.0)]

    assert test_near_atoms == ref_near_atoms


@pytest.mark.parametrize("method", [struc.CellList.get_atoms, struc.CellList.get_atoms_in_cells])
def test_empty_coordinates(method):
    """
    Test whether empty input coordinates result in an empty output
    array/mask.
    """
    array = strucio.load_structure(join(data_dir("structure"), "3o5r.bcif"))
    cell_list = struc.CellList(array, cell_size=10)

    indices = method(cell_list, np.array([]), 1, as_mask=False)
    mask = method(cell_list, np.array([]), 1, as_mask=True)
    assert len(indices) == 0
    assert len(mask) == 0
    assert indices.dtype == np.int32
    assert mask.dtype == bool


def test_pickle():
    """
    Test whether the CellList can be pickled and unpickled.
    The unpickled cell list should give the same results as the original cell list.
    """
    N_SAMPLES = 10
    CELL_SIZE = 5

    array = strucio.load_structure(join(data_dir("structure"), "3o5r.bcif"))
    original_cell_list = struc.CellList(array, cell_size=CELL_SIZE)
    pickled = pickle.dumps(original_cell_list)
    unpickled_cell_list = pickle.loads(pickled)
    assert np.array_equal(
        original_cell_list.get_atoms(array.coord[:N_SAMPLES], CELL_SIZE),
        unpickled_cell_list.get_atoms(array.coord[:N_SAMPLES], CELL_SIZE)
    )
