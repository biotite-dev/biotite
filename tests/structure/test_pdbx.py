# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

import biopython.structure as struc
import biopython.structure.io.pdbx as pdbx
import biopython.database.rcsb as rcsb
import biopython
import itertools
import numpy as np
import glob
from os.path import join
from .util import data_dir
import pytest


@pytest.mark.parametrize("path, is_stack, use_c_ext", itertools.product(
                            glob.glob(join(data_dir, "*.cif")),
                            [False, True],
                            [False, True],
                         ))
def test_conversion(path, is_stack, use_c_ext):
    if use_c_ext:
        if biopython.has_c_extensions():
            biopython.enable_c_extensions(True)
        else:
            return
    else:
        biopython.enable_c_extensions(False)
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

def test_extra_fields():
    path = join(data_dir, "1l2y.cif")
    pdbx_file = pdbx.PDBxFile()
    pdbx_file.read(path)
    stack1 = pdbx.get_structure(pdbx_file, extra_fields=["atom_id","b_factor",
                                "occupancy","charge"])
    pdbx_file = pdbx.PDBxFile()
    pdbx.set_structure(pdbx_file, stack1, data_block="test")
    stack2 = pdbx.get_structure(pdbx_file, extra_fields=["atom_id","b_factor",
                                "occupancy","charge"])
    assert stack1 == stack2

            
        
    