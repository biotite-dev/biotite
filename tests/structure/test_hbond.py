# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite.structure.hbond as hbond
import numpy as np
from biotite.structure.io import load_structure

def test_hbond_same_resid():
    """ 1ASN in 1l2y should form a hydrogen bond between its sidechain and the N-term """
    path = "tests/structure/data/1l2y.pdb"
    s = load_structure(path)

    triplets, mask = hbond.hbond(s[:, s.res_id == 1], s[:, s.res_id == 1])

    assert triplets[:, (triplets.res_id == 1) & (triplets.atom_name == 'N')].array_length() > 1

def test_hbond_total_count():
    """
    With the standart Baker & Hubbard criterion, 1l2y should have 28 hydrogen bonds with a frequency > 0.1
    (comparision with external calculation)
    """
    path = "tests/structure/data/1l2y.pdb"
    s = load_structure(path)

    triplets, mask = hbond.hbond(s)
    freq = hbond.get_hbond_frequency(mask)

    assert len(freq[freq >= 0.1]) == 28

def test_get_hbond_frequency():
    mask = np.array([
        [True, True, True, True, True], # 1.0
        [False, False, False, False, False], # 0.0
        [False, False, False, True, True] # 0.4
    ]).T
    print(hbond.get_hbond_frequency(mask))
    assert not np.isin(False, np.isclose(hbond.get_hbond_frequency(mask), np.array([1.0, 0.0, 0.4])))


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

