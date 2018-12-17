# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
import itertools
import numpy as np
import pytest
import biotite
import biotite.structure as struc
from biotite.structure.io import load_structure, save_structure
from .util import data_dir


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
@pytest.mark.xfail(raises=ImportError)
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