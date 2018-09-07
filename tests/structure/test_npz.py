# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import glob
from os.path import join, splitext
import numpy as np
import pytest
from pytest import approx
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.io.npz as npz
import biotite.structure.io.pdbx as pdbx
from .util import data_dir


@pytest.mark.parametrize("path", glob.glob(join(data_dir, "*.npz")))
def test_array_conversion(path):
    npz_file = npz.NpzFile()
    npz_file.read(path)
    array1 = npz_file.get_structure()
    npz_file = npz.NpzFile()
    npz_file.set_structure(array1)
    array2 = npz_file.get_structure()
    assert array1 == array2


@pytest.mark.parametrize("path", glob.glob(join(data_dir, "*.npz")))
def test_pdbx_consistency(path):
    cif_path = splitext(path)[0] + ".cif"
    array1 = strucio.load_structure(path)
    array2 = strucio.load_structure(cif_path)
    assert array1 == array2