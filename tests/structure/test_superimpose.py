# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.database.rcsb as rcsb
import biotite.structure as struc
import numpy as np
import glob
from os.path import join
from .util import data_dir
import pytest


@pytest.mark.parametrize("path", glob.glob(join(data_dir, "*.cif")))
def test_superimposition_array(path):
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

@pytest.mark.parametrize("ca_only", (True, False))
def test_superimposition_stack(ca_only):
    path = join(data_dir, "1l2y.cif")
    pdbx_file = pdbx.PDBxFile()
    pdbx_file.read(path)
    stack = pdbx.get_structure(pdbx_file)
    fixed = stack[0]
    mobile = stack[1:]
    fitted, transformation = struc.superimpose(fixed, mobile, ca_only)
    if ca_only:
        # The superimpositions are better for most cases than the
        # superimpositions in the structure file
        # -> Use average
        assert np.mean(struc.rmsd(fixed, fitted)) \
             < np.mean(struc.rmsd(fixed, mobile))
    else:
        # The superimpositions are better than the superimpositions
        # in the structure file
        assert (struc.rmsd(fixed, fitted) < struc.rmsd(fixed, mobile)).all()
