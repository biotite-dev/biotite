# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import biopython.structure as struc
import numpy as np
import pytest


def test_adjacency_map():
    array = struc.AtomArray(length=5)
    array.coord = np.array([[0,0,i] for i in range(5)])
    map = struc.AdjacencyMap(array, box_size=1)
    assert map.get_atoms([0,0,0.1], 1).tolist() == [0,1]
    assert map.get_atoms([0,0,1.1], 1).tolist() == [1,2]
    assert map.get_atoms([0,0,1.1], 2).tolist() == [0,1,2,3]