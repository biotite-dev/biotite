# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
import itertools
import numpy as np
import pytest
import biotite.structure as struc
from biotite.structure.io import load_structure
from ..util import data_dir, cannot_import


SAMPLE_BOXES = [
    (1, 1, 1,  90,  90,  90),
    (9, 5, 2,  90,  90,  90),
    (5, 5, 8,  90,  90, 120),
    (3, 2, 1,  10,  20,  20),
    (2, 4, 6, 100, 110, 120),
    (9, 9, 9,  90,  90, 170),
    (9, 8, 7,  50,  80,  50),
]

SAMPLE_COORD = [
    ( 1,  1,  1),
    ( 5, 10, 20),
    (-1,  5,  8),
    ( 3,  1, 54)
]



# Ignore warning about dummy unit cell vector
@pytest.mark.filterwarnings("ignore")
@pytest.mark.skipif(
    cannot_import("mdtraj"),
    reason="MDTraj is not installed"
)
@pytest.mark.parametrize(
    "len_a, len_b, len_c, alpha, beta, gamma", SAMPLE_BOXES
)
def test_box_vector_calculation(len_a, len_b, len_c, alpha, beta, gamma):
    box = struc.vectors_from_unitcell(
        len_a, len_b, len_c,
        np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma)
    )

    from mdtraj.utils import lengths_and_angles_to_box_vectors
    ref_box = np.stack(
        lengths_and_angles_to_box_vectors(
            len_a, len_b, len_c, alpha, beta, gamma
        )
    )
    assert np.allclose(box, ref_box)

    assert struc.unitcell_from_vectors(box) == pytest.approx(
        (len_a, len_b, len_c,
         alpha * 2*np.pi / 360, beta * 2*np.pi / 360, gamma * 2*np.pi / 360)
    )


def test_volume():
    # Very rudimentary test
    box = np.array([
        [5,0,0],
        [0,8,0],
        [0,0,2],
    ])
    assert struc.box_volume(box) == pytest.approx(80)
    boxes = np.stack([box, box])
    assert struc.box_volume(boxes) == pytest.approx(80,80)


@pytest.mark.parametrize(
    "len_a, len_b, len_c, alpha, beta, gamma, x, y,z",
    [box+coord for box, coord in itertools.product(SAMPLE_BOXES, SAMPLE_COORD)]
)
def test_move_into_box(len_a, len_b, len_c, alpha, beta, gamma, x, y, z):
    box = struc.vectors_from_unitcell(
        len_a, len_b, len_c,
        np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma)
    )
    coord = np.array([x,y,z])

    moved_coord = struc.move_inside_box(coord, box)
    fractions = struc.coord_to_fraction(moved_coord, box)
    assert ((fractions >= 0) & (fractions <=1)).all()


@pytest.mark.parametrize(
    "len_a, len_b, len_c, alpha, beta, gamma, x, y,z",
    [box+coord for box, coord in itertools.product(SAMPLE_BOXES, SAMPLE_COORD)]
)
def test_conversion_to_fraction(len_a, len_b, len_c,
                                alpha, beta, gamma,
                                x, y, z):
    box = struc.vectors_from_unitcell(
        len_a, len_b, len_c,
        np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma)
    )
    coord = np.array([x,y,z])

    fractions = struc.coord_to_fraction(coord, box)
    if struc.is_orthogonal(box):
        assert fractions.tolist() == pytest.approx(coord / np.diagonal(box))
    new_coord = struc.fraction_to_coord(fractions, box)
    assert np.allclose(coord, new_coord)

    coords = np.stack([coord, coord])
    boxes = np.stack([box, box])
    fractions = struc.coord_to_fraction(coords, boxes)
    new_coords = struc.fraction_to_coord(fractions, boxes)
    assert np.allclose(coords, new_coords)


def test_repeat_box():
    array = load_structure(join(data_dir("structure"), "3o5r.mmtf"))
    repeat_array, _ = struc.repeat_box(array)
    assert repeat_array[:array.array_length()] == array


def test_remove_pbc_unsegmented():
    """
    `remove_pbc()` should not alter unsegmented structures,
    when the structure is entirely in the box.
    Exclude the solvent, due to high distances between each atom. 
    """
    ref_array = load_structure(join(data_dir("structure"), "3o5r.mmtf"))
    # Center structure in box
    centroid = struc.centroid(ref_array)
    box_center = np.diag(ref_array.box) / 2
    ref_array = struc.translate(ref_array, box_center-centroid)
    # Remove solvent
    ref_array = ref_array[~struc.filter_solvent(ref_array)]
    array = struc.remove_pbc(ref_array)

    assert ref_array.equal_annotation_categories(array)
    assert np.allclose(ref_array.coord, array.coord)


@pytest.mark.parametrize(
    "multi_model, translation_vector",
    itertools.product(
        [False, True],
        [(20,30,40), (-11, 33, 22), (-40, -50, -60)]
    )
)
def test_remove_pbc_restore(multi_model, translation_vector):
    CUTOFF = 5.0
    
    def get_matrices(array):
        """
        Create a periodic and non-periodic adjacency matrix.
        """
        nonlocal CUTOFF
        if isinstance(array, struc.AtomArray):
            matrix     = struc.CellList(array, CUTOFF, periodic=False) \
                        .create_adjacency_matrix(CUTOFF)
            matrix_pbc = struc.CellList(array, CUTOFF, periodic=True) \
                        .create_adjacency_matrix(CUTOFF)
        elif isinstance(array, struc.AtomArrayStack):
            matrix = np.array(
                [struc.CellList(model, CUTOFF, periodic=False)
                 .create_adjacency_matrix(CUTOFF)
                 for model in array]
                )
            matrix_pbc = np.array(
                [struc.CellList(model, CUTOFF, periodic=True)
                 .create_adjacency_matrix(CUTOFF)
                 for model in array]
            )
        return matrix, matrix_pbc
    
    def assert_equal_matrices(array, matrix1, matrix2, periodic):
        """
        Due to numerical instability, entries in both matrices might
        be different, when the distance of atoms is almost equal to
        the cutoff distance of the matrix.
        This function checks, whether two atoms with unequal entries
        in the matrices are near the cutoff distance.
        """
        nonlocal CUTOFF
        indices = np.where(matrix1 != matrix2)
        for index in range(len(indices[0])):
            if len(indices) == 2:
                # multi_model = False -> AtomArray
                m = None
                i = indices[0][index]
                j = indices[1][index]
                box = array.box if periodic else None
                distance = struc.distance(array[i], array[j], box=box)
            if len(indices) == 3:
                # multi_model = True -> AtomArrayStack
                m = indices[0][index]
                i = indices[1][index]
                j = indices[2][index]
                box = array.box[m] if periodic else None
                distance = struc.distance(array[m,i], array[m,j], box=box)
            try:
                assert distance == pytest.approx(CUTOFF, abs=1e-4)
            except AssertionError:
                print(f"Model {m}, Atoms {i} and {j}")
                raise
    
    stack = load_structure(join(data_dir("structure"), "1gya.mmtf"))
    stack.box = np.array([
        np.diag(np.max(coord, axis=0) - np.min(coord, axis=0) + 10)
        for coord in stack.coord
    ])
    stack.coord -= np.min(stack.coord, axis=-2)[:, np.newaxis, :] - 5
    if multi_model:
        array = stack
    else:
        array = stack[0]

    # Use adjacency matrices instead of pairwise distances
    # for compuational efficiency
    ref_matrix, ref_matrix_pbc = get_matrices(array)

    array = struc.translate(array, translation_vector)
    array.coord = struc.move_inside_box(array.coord, array.box)
    moved_matrix, moved_matrix_pbc = get_matrices(array)
    # The translation and the periodic move should not
    # alter PBC-aware pairwise distances
    assert_equal_matrices(array, ref_matrix_pbc, moved_matrix_pbc, True)
    # Non-PBC-aware distances should change,
    # otherwise the atoms do not go over the periodic boundary
    # and the test does not make sense
    with pytest.raises(AssertionError):
        assert_equal_matrices(array, ref_matrix, moved_matrix, False)

    array = struc.remove_pbc(array)
    restored_matrix, restored_matrix_pbc = get_matrices(array)
    # Both adjacency matrices should be equal to the original ones,
    # as the structure should be completely restored
    assert_equal_matrices(array, ref_matrix_pbc, restored_matrix_pbc, True)
    assert_equal_matrices(array, ref_matrix,     restored_matrix,     False)


@pytest.mark.parametrize("multi_model", [True, False])
def test_remove_pbc_selections(multi_model):
    """
    This test makes no assertions, it only test whether an exception
    occurs, when the `selection` parameter is given in `remove_pbc()`.
    """
    array = load_structure(join(data_dir("structure"), "3o5r.mmtf"))
    if multi_model:
        array = struc.stack([array, array])
    
    struc.remove_pbc(array)
    struc.remove_pbc(array, array.chain_id[0])
    struc.remove_pbc(array, struc.filter_amino_acids(array))
    struc.remove_pbc(array, [struc.filter_amino_acids(array),
                       (array.res_name == "FK5")])
    # Expect error when selectinf an atom multiple times
    with pytest.raises(ValueError):
        struc.remove_pbc(array, [struc.filter_amino_acids(array),
                                 (array.atom_name == "CA")])