# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
import numpy as np
import pytest
import biotite
import biotite.structure as struc
from biotite.structure.io import load_structure, save_structure
from .util import data_dir


# Ignore warning about dummy unit cell vector
@pytest.mark.filterwarnings("ignore")
@pytest.mark.xfail(raises=ImportError)
@pytest.mark.parametrize("len_a, len_b, len_c, alpha, beta, gamma", [
    (1, 1, 1,  90,  90,  90),
    (9, 5, 2,  90,  90,  90),
    (5, 5, 8,  90,  90, 120),
    (3, 2, 1,  10,  20,  30),
    (2, 4, 6, 100, 110, 120),
    (9, 9, 9,  90,  90, 170),
    (9, 8, 7,  10,  20,  30),
])
def test_box_vector_calculation(len_a, len_b, len_c, alpha, beta, gamma):
    box = struc.vectors_from_unitcell(
        len_a, len_b, len_c,
        alpha * 2*np.pi / 360, beta * 2*np.pi / 360, gamma * 2*np.pi / 360
    )
    print(box)

    from mdtraj.utils import lengths_and_angles_to_box_vectors
    ref_box = np.stack(
        lengths_and_angles_to_box_vectors(
            len_a, len_b, len_c, alpha, beta, gamma
        )
    )
    print(ref_box)
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