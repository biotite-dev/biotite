# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

import biopython.structure as struc
import biopython.structure.io.npz as npz
import numpy as np
from os.path import join
from .util import data_dir
import pytest


def test_rotate_centered():
    file = npz.NpzFile()
    file.read(join(data_dir, "1l2y.npz"))
    array = file.get_structure()[0]
    rotated = struc.rotate_centered(array, [2*np.pi, 2*np.pi, 2*np.pi])
    assert np.sum(rotated.coord-array.coord) == pytest.approx(0)