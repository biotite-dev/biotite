# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import biopython.structure as struc
import biopython.structure.io.npz as npz
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

def test_get_residues(array):
    ids, names = struc.get_residues(array)
    assert ids.tolist() == list(range(1, 21))
    assert names.tolist() == ["ASN","LEU","TYR","ILE","GLN","TRP","LEU","LYS",
                              "ASP","GLY","GLY","PRO","SER","SER","GLY","ARG",
                              "PRO","PRO","PRO","SER"]