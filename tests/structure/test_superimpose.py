# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import biopython.structure as struc
import biopython.structure.io.pdbx as pdbx
import biopython.database.rcsb as rcsb
import biopython.structure as struc
import numpy as np
import glob
from os.path import join
from .util import data_dir
import pytest


@pytest.mark.parametrize("path", glob.glob(join(data_dir, "*.cif")))
def test_superimposition(path):
    pdbx_file = pdbx.PDBxFile()
    pdbx_file.read(path)
    fixed = pdbx.get_structure(pdbx_file, model=1)
    mobile = fixed.copy()
    mobile = struc.rotate(mobile, (1,2,3))
    mobile = struc.translate(mobile, (1,2,3))
    fitted, transformation = struc.superimpose(fixed, mobile, False)
    assert struc.rmsd(fixed, fitted) == pytest.approx(0)
    fitted = struc.superimpose_apply(mobile, transformation)
    assert struc.rmsd(fixed, fitted) == pytest.approx(0)