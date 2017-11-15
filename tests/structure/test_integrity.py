# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

import biotite.structure as struc
import biotite.structure.io.npz as npz
import numpy as np
from os.path import join
from .util import data_dir
import pytest


@pytest.fixture
def sample_array():
    file = npz.NpzFile()
    file.read(join(data_dir, "1l2y.npz"))
    return file.get_structure()[0]

@pytest.fixture
def gapped_sample_array(sample_array):
    sample_array = sample_array[sample_array.res_id != 5]
    sample_array = sample_array[(sample_array.res_id != 9) |
                                (sample_array.atom_name != "N")]
    return sample_array

def test_id_continuity_check(gapped_sample_array):
    discon = struc.check_id_continuity(gapped_sample_array)
    discon_array = gapped_sample_array[discon]
    assert discon_array.res_id.tolist() == [6]

def test_bond_continuity_check(gapped_sample_array):
    discon = struc.check_bond_continuity(gapped_sample_array)
    discon_array = gapped_sample_array[discon]
    assert discon_array.res_id.tolist() == [6,9]