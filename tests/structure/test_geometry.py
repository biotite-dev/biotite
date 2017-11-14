# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

import biotite.structure as struc
import biotite.structure.io.npz as npz
import numpy as np
from os.path import join
from .util import data_dir
import pytest


def test_distance():
    coord1 = struc.coord([0,1,1])
    coord2 = struc.coord([0,2,2])
    assert struc.distance(coord1, coord2) == pytest.approx(np.sqrt(2))

def test_centroid():
    coord = struc.coord([[1,1,1],[0,-1,-1],[-1,0,0]])
    assert struc.centroid(coord).tolist() == [0,0,0]

def test_angle():
    coord1 = struc.coord([0,0,1])
    coord2 = struc.coord([0,0,0])
    coord3 = struc.coord([0,1,1])
    assert struc.angle(coord1, coord2, coord3) == pytest.approx(0.25*np.pi)

def test_dihedral():
    coord1 = struc.coord([-0.5,-1,0])
    coord2 = struc.coord([0,0,0])
    coord3 = struc.coord([1,0,0])
    coord4 = struc.coord([0,0,-1])
    assert struc.dihedral(coord1, coord2, coord3, coord4) \
           == pytest.approx(0.5*np.pi)

def test_dihedral_backbone():
    file = npz.NpzFile()
    file.read(join(data_dir, "1l2y.npz"))
    array = file.get_structure()[0]
    phi, psi, omega = struc.dihedral_backbone(array, "A")
    # Remove nan values
    omega = np.abs(omega)[:-1]
    assert omega.tolist() == pytest.approx([np.pi] * len(omega),
                                                        rel=0.05)