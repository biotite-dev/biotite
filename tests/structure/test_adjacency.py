# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite.structure as struc
import numpy as np
import pytest


def test_adjacency_map():
    array = struc.AtomArray(length=5)
    array.coord = np.array([[0,0,i] for i in range(5)])
    map = struc.AdjacencyMap(array, box_size=1)
    assert map.get_atoms(np.array([0,0,0.1]), 1).tolist() == [0,1]
    assert map.get_atoms(np.array([0,0,1.1]), 1).tolist() == [1,2]
    assert map.get_atoms(np.array([0,0,1.1]), 2).tolist() == [0,1,2,3]