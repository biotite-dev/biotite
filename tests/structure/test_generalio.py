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
    if splitext(path)[1] in [".trr", ".xtc", ".tng", ".dcd", ".netcdf"]:
        template = strucio.load_structure(join(data_dir, "1l2y.mmtf"))
        array = strucio.load_structure(path, template)
    else:
        array = strucio.load_structure(path)


@pytest.mark.xfail(raises=ImportError)
def test_loading_template_with_trj():
    template = join(data_dir, "1l2y.pdb")
    trajectory = join(data_dir, "1l2y.xtc")
    stack = strucio.load_structure(trajectory, template)
    assert isinstance(stack, struc.AtomArrayStack)
    assert len(stack) > 1


@pytest.mark.xfail(raises=ImportError)
def test_loading_with_extra_args():
    template = join(data_dir, "1l2y.pdb")
    trajectory = join(data_dir, "1l2y.xtc")

    # test if arguments are passed to text files as get_structure arg
    structure = strucio.load_structure(template, extra_fields=["b_factor"])
    assert "b_factor" in structure.get_annotation_categories()

    # test if arguments are passed to read for trajectories
    stack = strucio.load_structure(trajectory, template=struc[0], start=5, stop=6)
    assert len(stack) == 1

    # loading should fail with wrong arguments
    with pytest.raises(TypeError):
        strucio.load_structure(template, start=2)
    

@pytest.mark.xfail(raises=ImportError)
@pytest.mark.parametrize(
    "suffix",
    ["pdb", "cif", "gro", "pdbx", "mmtf",
     "trr", "xtc", "tng", "dcd", "netcdf"]
)
def test_saving(suffix):
    array = strucio.load_structure(join(data_dir, "1l2y.mmtf"))
    strucio.save_structure(
        biotite.temp_file("1l2y." + suffix), array
    )


@pytest.mark.xfail(raises=ImportError)
@pytest.mark.parametrize(
    "suffix",
    ["pdb", "cif", "gro", "pdbx", "mmtf",
     "trr", "xtc", "tng", "dcd", "netcdf"]
)
def test_saving_with_extra_args(suffix):
    array = strucio.load_structure(join(data_dir, "1l2y.mmtf"))
    with pytest.raises(TypeError):
        strucio.save_structure(
            biotite.temp_file("1l2y." + suffix), array, answer=42
        )
