# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite
import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np
import glob
import itertools
from os.path import join, basename, splitext
from .util import data_dir
import pytest


@pytest.mark.xfail(raises=ImportError)
@pytest.mark.parametrize("path", glob.glob(join(data_dir, "1l2y.*")))
def test_loading(path):
    if splitext(path)[1] in [".trr", ".xtc", ".tng"]:
        template = strucio.load_structure(join(data_dir, "1l2y.mmtf"))
        array = strucio.load_structure(path, template)
    else:
        array = strucio.load_structure(path)


@pytest.mark.parametrize("suffix", ["pdb","cif","pdbx","mmtf"])
def test_saving(suffix):
    array = strucio.load_structure(join(data_dir, "1l2y.mmtf"))
    strucio.save_structure(biotite.temp_file("1l2y." + suffix),
                           array)