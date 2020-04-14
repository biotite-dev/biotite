# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import glob
import itertools
from os.path import join, splitext
import pytest
from pytest import approx
import numpy as np
import biotite
import biotite.structure.io as io
import biotite.structure.io.gro as gro
import biotite.structure.io.pdb as pdb
from biotite.structure import Atom, array
from ..util import data_dir


@pytest.mark.parametrize(
    "path, single_model",
    itertools.product(
        glob.glob(join(data_dir("structure"), "*.gro")),
        [False, True]
    )
)
def test_array_conversion(path, single_model):
    model = 1 if single_model else None
    gro_file = gro.GROFile()
    gro_file.read(path)
    array1 = gro_file.get_structure(model=model)
    gro_file = gro.GROFile()
    gro_file.set_structure(array1)
    array2 = gro_file.get_structure(model=model)
    if array1.box is not None:
        assert np.allclose(array1.box, array2.box)
    assert array1.bonds == array2.bonds
    for category in array1.get_annotation_categories():
        assert array1.get_annotation(category).tolist() == \
               array2.get_annotation(category).tolist()
    assert array1.coord.tolist() == array2.coord.tolist()


@pytest.mark.parametrize(
    "path", glob.glob(join(data_dir("structure"), "[!(waterbox)]*.gro"))
)
def test_pdb_consistency(path):
    pdb_path = splitext(path)[0] + ".pdb"
    pdb_file = pdb.PDBFile()
    pdb_file.read(pdb_path)
    a1 = pdb_file.get_structure(model=1)
    gro_file = gro.GROFile()
    gro_file.read(path)
    a2 = gro_file.get_structure(model=1)

    assert a1.array_length() == a2.array_length()

    for category in ["res_id", "res_name", "atom_name"]:
        assert a1.get_annotation(category).tolist() == \
               a2.get_annotation(category).tolist()

    # Mind rounding errors when converting pdb to gro (A -> nm)
    assert a1.coord.flatten().tolist() \
        == approx(a2.coord.flatten().tolist(), abs=1e-2)


@pytest.mark.parametrize(
    "path, single_model",
    itertools.product(
        glob.glob(join(data_dir("structure"), "*.pdb")),
        [False, True]
    )
)
def test_pdb_to_gro(path, single_model):
    # Converting stacks between formats should not change data
    model = 1 if single_model else None
    
    # Read in data
    pdb_file = pdb.PDBFile()
    pdb_file.read(path)
    a1 = pdb_file.get_structure(model=model)

    # Save stack as gro
    tmp_file_name = biotite.temp_file("gro")
    gro_file = gro.GROFile()
    gro_file.set_structure(a1)
    gro_file.write(tmp_file_name)

    # Reload stack from gro
    gro_file = gro.GROFile()
    gro_file.read(tmp_file_name)
    a2 = gro_file.get_structure(model=model)

    assert a1.array_length() == a2.array_length()

    for category in ["res_id", "res_name", "atom_name"]:
        assert a1.get_annotation(category).tolist() == \
               a2.get_annotation(category).tolist()

    # Mind rounding errors when converting pdb to gro (A -> nm)
    assert a1.coord.flatten().tolist() \
        == approx(a2.coord.flatten().tolist(), abs=1e-2)


def test_gro_id_overflow():
    # Create an oversized AtomArray where atom_id > 100000 and res_id > 10000
    num_atoms = 100005
    atoms = array([Atom([1,2,3], atom_name="CA", element="C", res_name="X",
                        res_id=i+1) for i in range(num_atoms)])
    atoms.box = np.array([[1,0,0], [0,1,0], [0,0,1]])

    # Write .gro file
    tmp_file_name = biotite.temp_file(".gro")
    io.save_structure(tmp_file_name, atoms)

    # Read .gro file
    gro_file = gro.GROFile()
    gro_file.read(tmp_file_name)
    s = gro_file.get_structure()

    assert s.array_length() == num_atoms


def test_gro_no_box():
    """
    .gro file format requires valid box parameters at the end of each
    model. However, if we read such a file in, the resulting object should not
    have an assigned box.
    """

    # Create an AtomArray
    atom = Atom([1,2,3], atom_name="CA", element="C", res_name="X", res_id=1)
    atoms = array([atom])

    # Write .gro file
    tmp_file_name = biotite.temp_file(".gro")
    io.save_structure(tmp_file_name, atoms)
    
    # Read in file
    gro_file = gro.GROFile()
    gro_file.read(tmp_file_name)
    s = gro_file.get_structure()

    # Assert no box with 0 dimension
    assert s.box is None