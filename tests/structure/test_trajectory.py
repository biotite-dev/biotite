# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.io.xtc as xtc
import biotite.structure.io.trr as trr
import biotite.structure.io.pdbx as pdbx
import numpy as np
import glob
from os.path import join, basename
from .util import data_dir
import pytest


@pytest.mark.xfail(raises=ImportError)
@pytest.mark.parametrize("format", ["trr","xtc"])
def test_array_conversion(format):
    template = strucio.load_structure(join(data_dir, "1l2y.mmtf"))[0]
    # Add fake box
    template.box = np.diag([1,2,3])
    if format == "trr":
        traj_file_cls = trr.TRRFile
    if format == "xtc":
        traj_file_cls = xtc.XTCFile
    
    traj_file = traj_file_cls()
    traj_file.read(join(data_dir, f"1l2y.{format}"))
    ref_array = traj_file.get_structure(template)

    traj_file = traj_file_cls()
    traj_file.set_structure(ref_array)
    file_name = biotite.temp_file(format)
    traj_file.write(file_name)

    traj_file = traj_file_cls()
    traj_file.read(file_name)
    array = traj_file.get_structure(template)
    assert ref_array.bonds == array.bonds
    assert ref_array.equal_annotation_categories(array)
    assert ref_array.box == pytest.approx(array.box)
    assert ref_array.coord == pytest.approx(array.coord, abs=1e-2)


@pytest.mark.xfail(raises=ImportError)
@pytest.mark.parametrize("format", ["trr","xtc"])
def test_PDBx_consistency(format):
    ref_array = strucio.load_structure(join(data_dir, "1l2y.cif"))
    # Only first model
    template = strucio.load_structure(join(data_dir, "1l2y.mmtf"))[0]
    if format == "trr":
        traj_file_cls = trr.TRRFile
    if format == "xtc":
        traj_file_cls = xtc.XTCFile
    traj_file = traj_file_cls()
    traj_file.read(join(data_dir, f"1l2y.{format}"))
    array = traj_file.get_structure(template)
    # 1l2y has no box
    # assert np.array_equal(array1.box, array2.box)
    assert ref_array.bonds == array.bonds
    assert ref_array.equal_annotation_categories(array)
    assert ref_array.coord == pytest.approx(array.coord)