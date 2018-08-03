# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite.structure as struc
import numpy as np
import pytest


def test_cell_list():
    array = struc.AtomArray(length=5)
    array.coord = np.array([[0,0,i] for i in range(5)])
    cell_list = struc.CellList(array, cell_size=1)
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