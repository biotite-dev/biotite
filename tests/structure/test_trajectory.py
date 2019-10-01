# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
import glob
from os.path import join, basename
import numpy as np
import pytest
import biotite
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.io.xtc as xtc
import biotite.structure.io.trr as trr
import biotite.structure.io.tng as tng
import biotite.structure.io.dcd as dcd
import biotite.structure.io.netcdf as netcdf
import biotite.structure.io.pdbx as pdbx
from .util import data_dir


@pytest.mark.xfail(raises=ImportError)
@pytest.mark.parametrize("format", ["trr", "xtc", "tng", "dcd", "netcdf"])
def test_array_conversion(format):
    template = strucio.load_structure(join(data_dir, "1l2y.mmtf"))[0]
    # Add fake box
    template.box = np.diag([1,2,3])
    if format == "trr":
        traj_file_cls = trr.TRRFile
    if format == "xtc":
        traj_file_cls = xtc.XTCFile
    if format == "tng":
        traj_file_cls = tng.TNGFile
    if format == "dcd":
        traj_file_cls = dcd.DCDFile
    if format == "netcdf":
        traj_file_cls = netcdf.NetCDFFile
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
@pytest.mark.parametrize(
    "format, start, stop, chunk_size",
    itertools.product(
        ["trr", "xtc", "tng", "dcd", "netcdf"],
        [None, 2],
        [None, 17],
        [None, 3]
    )
)
def test_mmtf_consistency(format, start, stop, chunk_size):
    # MMTF is used as reference for consistency check
    # due to higher performance
    ref_traj = strucio.load_structure(join(data_dir, "1l2y.mmtf"))
    ref_traj = ref_traj[slice(start, stop)]
    
    # Template is first model of the reference
    template = ref_traj[0]
    if format == "trr":
        traj_file_cls = trr.TRRFile
    if format == "xtc":
        traj_file_cls = xtc.XTCFile
    if format == "tng":
        traj_file_cls = tng.TNGFile
    if format == "dcd":
        traj_file_cls = dcd.DCDFile
    if format == "netcdf":
        traj_file_cls = netcdf.NetCDFFile
    traj_file = traj_file_cls()
    traj_file.read(
        join(data_dir, f"1l2y.{format}"),
        start, stop, chunk_size=chunk_size
    )
    test_traj = traj_file.get_structure(template)
    
    # 1l2y has no box
    # assert np.array_equal(test_traj.box, ref_traj.box)
    assert test_traj.bonds == ref_traj.bonds
    assert test_traj.equal_annotation_categories(ref_traj)
    assert test_traj.coord == pytest.approx(ref_traj.coord, abs=1e-2)