# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
import itertools
import warnings
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.io.mmtf as mmtf
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


@pytest.mark.parametrize("multi_model", [False, True])
def test_repeat_box(multi_model):
    model = None if multi_model else 1
    array = mmtf.get_structure(
        mmtf.MMTFFile.read(join(data_dir("structure"), "3o5r.mmtf")),
        model=model, include_bonds=True
    )
    repeat_array, _ = struc.repeat_box(array)
    assert repeat_array.array_length() == array.array_length() * 27
    assert repeat_array[..., :array.array_length()] == array


@pytest.mark.parametrize("multi_model", [True, False])
def test_remove_pbc_unsegmented(multi_model):
    """
    `remove_pbc()` should not alter unsegmented structures,
    when the structure is entirely in the box.
    """
    model = None if multi_model else 1
    ref_array = load_structure(
        join(data_dir("structure"), "3o5r.mmtf"),
        model=model,
        include_bonds=True
    )
    # Center structure in box
    centroid = struc.centroid(ref_array)
    box_center = np.diag(ref_array.box) / 2
    ref_array = struc.translate(ref_array, box_center-centroid)
    test_array = struc.remove_pbc(ref_array)

    assert ref_array.equal_annotation_categories(test_array)
    assert np.allclose(ref_array.coord, test_array.coord)


@pytest.mark.parametrize(
    "multi_model, seed",
    itertools.product(
        [False, True],
        range(10)
    )
)
def test_remove_pbc_restore(multi_model, seed):
    BUFFER = 5

    def get_distance_matrices(array):
        if isinstance(array, struc.AtomArray):
            matrix = struc.distance(
                array.coord[:, np.newaxis, :],
                array.coord[np.newaxis, :, :],
                box=None
            )
            matrix_pbc = struc.distance(
                array.coord[:, np.newaxis, :],
                array.coord[np.newaxis, :, :],
                box=array.box
            )
        elif isinstance(array, struc.AtomArrayStack):
            matrices = [get_distance_matrices(model) for model in array]
            matrix = np.stack([m[0] for m in matrices])
            matrix_pbc = np.stack([m[1] for m in matrices])
        return matrix, matrix_pbc
    
    stack = load_structure(
        join(data_dir("structure"), "1l2y.mmtf"), include_bonds=True
    )
    
    # Only consider a single molecule
    # -> remove all other atoms (in this case some unbound hydrogen)
    mol_masks = struc.get_molecule_masks(stack)
    largest_mask = mol_masks[np.argmax(np.count_nonzero(mol_masks, axis=-1))]
    stack = stack[..., largest_mask]

    # Create a relatively tight box around the protein
    stack.box = np.array([
        np.diag(np.max(coord, axis=0) - np.min(coord, axis=0) + BUFFER)
        for coord in stack.coord
    ])
    stack.coord -= np.min(stack.coord, axis=-2)[:, np.newaxis, :] + BUFFER / 2
    if multi_model:
        array = stack
    else:
        array = stack[0]
    ref_matrix, ref_matrix_pbc = get_distance_matrices(array)

    ## Segment protein
    np.random.seed(seed)
    size = (array.stack_depth(), 3) if isinstance(array, struc.AtomArrayStack) else 3
    translation_vector = np.sum(
        np.random.uniform(-5, 5, size)[:, np.newaxis] * array.box,
        axis=-2
    )[..., np.newaxis, :]
    # Move atoms over periodic boundary...
    array = struc.translate(array, translation_vector)
    # ... and enforce segmentation
    array.coord = struc.move_inside_box(array.coord, array.box)
    # Check if segmentation was achieved
    # This is a prerequisite to properly test 'remove_pbc()'
    segment_matrix, segment_matrix_pbc = get_distance_matrices(array)
    assert not np.allclose(segment_matrix, ref_matrix, atol=1e-4)
    assert np.allclose(segment_matrix_pbc, ref_matrix_pbc, atol=1e-4)

    ## Remove segmentation again via 'remove_pbc()'
    array = struc.remove_pbc(array)
    # Now the pairwise atom distances should be the same
    # as in the beginning again
    test_matrix, test_matrix_pbc = get_distance_matrices(array)
    assert np.allclose(test_matrix, ref_matrix, atol=1e-4)
    assert np.allclose(test_matrix_pbc, ref_matrix_pbc, atol=1e-4)

    # The centroid of the structure should be inside the box dimensions
    centroid = struc.centroid(array)
    assert np.all(
        (centroid > np.zeros(3)) &
        (centroid < np.sum(array.box, axis=-2))
    )


@pytest.mark.parametrize("multi_model", [True, False])
def test_remove_pbc_selection(multi_model):
    """
    This test makes no assertions, it only test whether an exception
    occurs, when the `selection` parameter is given in `remove_pbc()`.
    """
    array = load_structure(join(data_dir("structure"), "3o5r.mmtf"))
    if multi_model:
        array = struc.stack([array, array])
    
    select_all = np.ones(array.array_length(), dtype=bool)
    select_none = np.zeros(array.array_length(), dtype=bool)
    assert struc.remove_pbc(array, select_all) == struc.remove_pbc(array)
    with warnings.catch_warnings():
        # A warning due to a zero-division (centroid of empty list of
        # atoms) is raised here
        warnings.simplefilter("ignore")
        assert struc.remove_pbc(array, select_none) == array