# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import biopython.structure as struc
import biopython.structure.io.pdbx as pdbx
import biopython.database.rcsb as rcsb
import numpy as np
import glob
from os.path import join
from .util import data_dir
import pytest


@pytest.mark.parametrize("path", glob.glob(join(data_dir, "*.cif")))
def test_array_conversion(path):
    _test_conversion(path, is_stack=False)

@pytest.mark.parametrize("path", glob.glob(join(data_dir, "*.cif")))
def test_stack_conversion(path):
    _test_conversion(path, is_stack=True)
        
def _test_conversion(path, is_stack):
    pdbx_file = pdbx.PDBxFile()
    pdbx_file.read(path)
    if is_stack:
        array1 = pdbx.get_structure(pdbx_file)
    else:
        array1 = pdbx.get_structure(pdbx_file, model=1)
    pdbx_file = pdbx.PDBxFile()
    pdbx.set_structure(pdbx_file, array1, data_block="test")
    if is_stack:
        array2 = pdbx.get_structure(pdbx_file)
    else:
        array2 = pdbx.get_structure(pdbx_file, model=1)
    assert array1 == array2
            
        
    