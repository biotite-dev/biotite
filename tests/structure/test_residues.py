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
def array():
    file = npz.NpzFile()
    file.read(join(data_dir, "1l2y.npz"))
    return file.get_structure()[0]

def test_apply_residue_wise(array):
    data = struc.apply_residue_wise(array, np.ones(len(array)), np.sum)
    assert data.tolist() == [len(array[array.res_id == i])
                             for i in range(1, 21)]

def test_spread_residue_wise(array):
    input_data = np.arange(1,21)
    output_data = struc.spread_residue_wise(array, input_data)
    assert output_data.tolist() == array.res_id.tolist()

def test_get_residues(array):
    ids, names = struc.get_residues(array)
    assert ids.tolist() == list(range(1, 21))
    assert names.tolist() == ["ASN","LEU","TYR","ILE","GLN","TRP","LEU","LYS",
                              "ASP","GLY","GLY","PRO","SER","SER","GLY","ARG",
                              "PRO","PRO","PRO","SER"]