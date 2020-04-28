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
from ..util import data_dir, cannot_import


@pytest.mark.skipif(
    cannot_import("mdtraj"),
    reason="MDTraj is not installed"
)
@pytest.mark.parametrize("format", ["trr", "xtc", "tng", "dcd", "netcdf"])
def test_array_conversion(format):
    template = strucio.load_structure(
        join(data_dir("structure"), "1l2y.mmtf")
    )[0]
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
    traj_file = traj_file_cls.read(
        join(data_dir("structure"), f"1l2y.{format}")
    )
    ref_array = traj_file.get_structure(template)

    traj_file = traj_file_cls()
    traj_file.set_structure(ref_array)
    file_name = biotite.temp_file(format)
    traj_file.write(file_name)

    traj_file = traj_file_cls.read(file_name)
    array = traj_file.get_structure(template)
    assert ref_array.bonds == array.bonds
    assert ref_array.equal_annotation_categories(array)
    assert ref_array.box == pytest.approx(array.box)
    assert ref_array.coord == pytest.approx(array.coord, abs=1e-2)


@pytest.mark.skipif(
    cannot_import("mdtraj"),
    reason="MDTraj is not installed"
)
@pytest.mark.parametrize(
    "format, start, stop, step, chunk_size",
    itertools.product(
        ["trr", "xtc", "tng", "dcd", "netcdf"],
        [None, 2],
        [None, 17],
        [None, 2],
        [None, 3]
    )
)
def test_mmtf_consistency(format, start, stop, step, chunk_size):
    if format == "netcdf" and stop is not None and step is not None:
        # Currently, there is an inconsistency in in MDTraj's
        # NetCDFTrajectoryFile class:
        # In this class the number of frames in the output arrays
        # is dependent on the 'stride' parameter
        return
    
    # MMTF is used as reference for consistency check
    # due to higher performance
    ref_traj = strucio.load_structure(join(data_dir("structure"), "1l2y.mmtf"))
    ref_traj = ref_traj[slice(start, stop, step)]
    
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
    traj_file = traj_file_cls.read(
        join(data_dir("structure"), f"1l2y.{format}"),
        start, stop, step, chunk_size=chunk_size
    )
    test_traj = traj_file.get_structure(template)
    test_traj_time = traj_file.get_time()
    
    if format not in ["dcd", "netcdf"]:
        # The time starts at 1.0 and increases by 1.0 each step
        # -> can be tested against 'range()' function
        # Shift to ensure time starts at 0
        test_traj_time -= 1
        start = start if start is not None else 0
        stop = stop if stop is not None else 38     # 38 models in 1l2y
        step = step if step is not None else 1
        assert test_traj_time.astype(int).tolist() \
            == list(range(start, stop, step))

    assert test_traj.stack_depth() == ref_traj.stack_depth()
    # 1l2y has no box
    # no assert np.array_equal(test_traj.box, ref_traj.box)
    assert test_traj.bonds == ref_traj.bonds
    assert test_traj.equal_annotation_categories(ref_traj)
    assert test_traj.coord == pytest.approx(ref_traj.coord, abs=1e-2)


@pytest.mark.skipif(
    cannot_import("mdtraj"),
    reason="MDTraj is not installed"
)
@pytest.mark.parametrize(
    "format, start, stop, step",
    itertools.product(
        ["trr", "xtc", "tng", "dcd", "netcdf"],
        [None, 2],
        [None, 17],
        [None, 2]
    )
)
def test_read_iter(format, start, stop, step):
    """
    Compare aggregated yields of :func:`read_iter()` with the values
    from a corresponding :class:`TrajectoryFile` object.
    """
    if format == "netcdf" and step is not None:
        # Currently, there is an inconsistency in in MDTraj's
        # NetCDFTrajectoryFile class:
        # In this class the number of frames in the output arrays
        # is dependent on the 'stride' parameter
        return
    
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
    file_name = join(data_dir("structure"), f"1l2y.{format}")
    
    traj_file = traj_file_cls.read(file_name, start, stop, step)
    ref_coord = traj_file.get_coord()
    ref_box = traj_file.get_box()
    ref_time = traj_file.get_time()

    test_coord = []
    test_box = []
    test_time = []
    for coord, box, time in traj_file.read_iter(file_name, start, stop, step):
        test_coord.append(coord)
        test_box.append(box)
        test_time.append(time)
    # Convert list to NumPy array
    test_coord = np.stack(test_coord)
    test_box = np.stack(test_box)
    test_time = np.stack(test_time)

    assert test_coord.tolist() == ref_coord.tolist()
    
    if ref_box is None:
        assert (test_box == None).all()
    else:
        assert test_box.tolist() == ref_box.tolist()
    
    if ref_time is None:
        assert (test_time == None).all()
    else:
        assert test_time.tolist() == ref_time.tolist()


@pytest.mark.skipif(
    cannot_import("mdtraj"),
    reason="MDTraj is not installed"
)
@pytest.mark.parametrize(
    "format, start, stop, step",
    itertools.product(
        ["trr", "xtc", "tng", "dcd", "netcdf"],
        [None, 2],
        [None, 17],
        [None, 2]
    )
)
def test_read_iter_structure(format, start, stop, step):
    """
    Compare aggregated yields of :func:`read_iter_structure()` with the
    return value of :func:`get_structure()` from a corresponding
    :class:`TrajectoryFile` object.
    """
    if format == "netcdf" and step is not None:
        # Currently, there is an inconsistency in in MDTraj's
        # NetCDFTrajectoryFile class:
        # In this class the number of frames in the output arrays
        # is dependent on the 'stride' parameter
        return
    
    template = strucio.load_structure(join(data_dir("structure"), "1l2y.mmtf"))
    
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
    file_name = join(data_dir("structure"), f"1l2y.{format}")
    
    traj_file = traj_file_cls.read(file_name, start, stop, step)
    ref_traj = traj_file.get_structure(template)
    
    test_traj = struc.stack(
        [frame for frame in traj_file_cls.read_iter_structure(
            file_name, template, start, stop, step
        )]
    )
    
    assert test_traj == ref_traj