# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from tempfile import NamedTemporaryFile
import biotite.structure as struc
import biotite.structure.io as strucio
from biotite.structure.io.general import _guess_element
import numpy as np
import glob
import os
import itertools
from os.path import join, basename, splitext
from ..util import data_dir, cannot_import
import pytest


@pytest.mark.skipif(
    cannot_import("mdtraj"),
    reason="MDTraj is not installed"
)
@pytest.mark.parametrize(
    "path", glob.glob(join(data_dir("structure"), "1l2y.*"))
)
def test_loading(path):
    """
    Just check if :func:`load_structure()` does not raise an exception
    and returns an object of appropriate type.
    """
    suffix = splitext(path)[1]
    if suffix in [".trr", ".xtc", ".tng", ".dcd", ".netcdf"]:
        template = strucio.load_structure(
            join(data_dir("structure"), "1l2y.mmtf")
        )
        array = strucio.load_structure(path, template)
    else:
        array = strucio.load_structure(path)
    if suffix == ".gro":
        # The test GRO file contains only a single model,
        # since it is created by Gromacs
        assert isinstance(array, struc.AtomArray)
    else:
        assert isinstance(array, struc.AtomArrayStack)


@pytest.mark.skipif(
    cannot_import("mdtraj"),
    reason="MDTraj is not installed"
)
def test_loading_template_with_trj():
    """
    Check if :func:`load_structure()` using a trajectory file does not
    raise an exception and returns an object of appropriate type and
    shape.
    """
    template = join(data_dir("structure"), "1l2y.pdb")
    trajectory = join(data_dir("structure"), "1l2y.xtc")
    stack = strucio.load_structure(trajectory, template)
    assert isinstance(stack, struc.AtomArrayStack)
    assert len(stack) > 1


@pytest.mark.skipif(
    cannot_import("mdtraj"),
    reason="MDTraj is not installed"
)
def test_loading_with_extra_args():
    """
    Check if :func:`load_structure()` witt optional arguments does not
    raise an exception and returns an object of appropriate type.
    """
    template = join(data_dir("structure"), "1l2y.pdb")
    trajectory = join(data_dir("structure"), "1l2y.xtc")

    # test if arguments are passed to text files as get_structure arg
    structure = strucio.load_structure(template, extra_fields=["b_factor"])
    assert "b_factor" in structure.get_annotation_categories()

    # test if arguments are passed to read for trajectories
    stack = strucio.load_structure(
        trajectory, template=structure[0], start=5, stop=6
    )
    assert len(stack) == 1

    # loading should fail with wrong arguments
    with pytest.raises(TypeError):
        strucio.load_structure(template, start=2)
    
    # test if atom_i argument is passed to templates
    stack = strucio.load_structure(trajectory, template, atom_i=[1, 2])
    assert stack.shape[1] == 2


@pytest.mark.skipif(
    cannot_import("mdtraj"),
    reason="MDTraj is not installed"
)
@pytest.mark.parametrize(
    "suffix",
    ["pdb", "cif", "gro", "pdbx", "mmtf",
     "trr", "xtc", "tng", "dcd", "netcdf"]
)
def test_saving(suffix):
    """
    Check if loading a structure from a file written via
    :func:`save_structure()` gives the same result as the input to
    :func:`save_structure()`.
    """
    path = join(data_dir("structure"), "1l2y.mmtf")
    ref_array = strucio.load_structure(path)
    if suffix in ("trr", "xtc", "tng", "dcd", "netcdf"):
        # Reading a trajectory file requires a template
        template = path
    else:
        template = None

    temp = NamedTemporaryFile("w", suffix=f".{suffix}", delete=False)
    strucio.save_structure(temp.name, ref_array)
    temp.close()
    
    test_array = strucio.load_structure(temp.name, template)
    os.remove(temp.name)

    for category in ref_array.get_annotation_categories():
        if category == "chain_id" and suffix == "gro":
            # The chain ID is not written to GRO files
            continue
        assert test_array.get_annotation(category).tolist() \
            ==  ref_array.get_annotation(category).tolist()
    assert test_array.coord.flatten().tolist() == pytest.approx(
            ref_array.coord.flatten().tolist(), abs=1e-2
    )


@pytest.mark.skipif(
    cannot_import("mdtraj"),
    reason="MDTraj is not installed"
)
@pytest.mark.parametrize(
    "suffix",
    ["pdb", "cif", "gro", "pdbx", "mmtf",
     "trr", "xtc", "tng", "dcd", "netcdf"]
)
def test_saving_with_extra_args(suffix):
    """
    Test if giving a wrong optional parameter to
    :func:`save_structure()` raises a :class:`TypeError`
    """
    array = strucio.load_structure(join(data_dir("structure"), "1l2y.mmtf"))
    temp = NamedTemporaryFile("w+", suffix=f".{suffix}")
    with pytest.raises(TypeError):
        strucio.save_structure(
            temp.name, array, answer=42
        )
    temp.close()


def test_small_molecule():
    """
    Check if loading a small molecule file written via
    :func:`save_structure()` gives the same result as the input to
    :func:`save_structure()`.
    """
    path = join(data_dir("structure"), "molecules", "TYR.sdf")
    ref_array = strucio.load_structure(path)
    temp = NamedTemporaryFile("w", suffix=".sdf", delete=False)
    strucio.save_structure(temp.name, ref_array)
    temp.close()
    
    test_array = strucio.load_structure(temp.name)
    os.remove(temp.name)

    assert test_array == ref_array


@pytest.mark.parametrize(
    "name,expected",
    [("CA", "C"),
     ("C", "C"),
     ("CB", "C"),
     ("OD1", "O"),
     ("HD21", "H"),
     ("1H", "H"),
     ("CL", "C"),
     ("HE", "H"),
     ("SD", "S"),
     ("NA", "N"),
     ("NX", "N"),
     ("BE", "BE"),
     ("BEA", "BE"),
     ("K", "K"),
     ("KA", "K"),
     ("QWERT", "")]
)
def test_guess_element(name, expected):
    """
    Check if elements are correctly guessed based on known examples.
    Elements are automatically guessed in GRO and PDB files where the
    *element* column is missing.
    """
    result = _guess_element(name)
    assert result == expected
