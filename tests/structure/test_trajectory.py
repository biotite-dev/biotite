# Copyright 2018 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License. Please see 'LICENSE.rst' for further information.

import biotite.structure as struc
import biotite.structure.io.xtc as xtc
import biotite.structure.io.trr as trr
import biotite.structure.io.pdbx as pdbx
import numpy as np
import glob
from os.path import join, basename
from .util import data_dir
import pytest


@pytest.mark.parametrize("format", ["trr","xtc"])
def test_PDBx_consistency(format):
    pdbx_file = pdbx.PDBxFile()
    pdbx_file.read(join(data_dir, "1l2y.cif"))
    array1 = pdbx.get_structure(pdbx_file)
    template = pdbx.get_structure(pdbx_file, model=1)
    if format == "trr":
        traj_file = trr.TRRFile()
        traj_file.read(join(data_dir, "1l2y.trr"))
    if format == "xtc":
        traj_file = xtc.XTCFile()
        traj_file.read(join(data_dir, "1l2y.xtc"))
    array2 = traj_file.get_structure(template)
    for cat in array1. get_annotation_categories():
        assert array1.get_annotation(cat).tolist() == \
               array2.get_annotation(cat).tolist()
        print(array1.coord[0,0], array2.coord[0,0])
        assert array1.coord == pytest.approx(array2.coord)