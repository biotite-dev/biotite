# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite.structure.hbond as hbond
import numpy as np

def test_hbond():
    struct = "tests/structure/data/"

def test_is_hbond():
    hbond_coords_valid = np.array([
        [1.0, 1.0, 0.0],  # donor
        [1.0, 2.0, 0.0],  # donor_h
        [1.0, 3.0, 0.0]  # acceptor
    ])

    hbond_coords_wrong_angle = np.array([
        [1.0, 1.0, 0.0],  # donor
        [1.0, 2.0, 0.0],  # donor_h
        [2.0, 2.0, 0.0]   # acceptor
    ])


    hbond_coords_wrong_distance = np.array([
        [1.0, 1.0, 0.0],  # donor
        [1.0, 2.0, 0.0],  # donor_h
        [4.0, 3.0, 0.0]  # acceptor
    ])

    assert hbond.is_hbond(*hbond_coords_valid)
    assert not hbond.is_hbond(*hbond_coords_wrong_angle)
    assert not hbond.is_hbond(*hbond_coords_wrong_distance)

