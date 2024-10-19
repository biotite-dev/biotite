# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import glob
import itertools
from os.path import join, splitext
from tempfile import TemporaryFile
import numpy as np
import pytest
from pytest import approx
import biotite
import biotite.structure.io.gro as gro
import biotite.structure.io.pdb as pdb
from biotite.structure import Atom, array
from tests.util import data_dir


def test_get_model_count():
    gro_file = gro.GROFile.read(join(data_dir("structure"), "1l2y.gro"))
    test_model_count = gro_file.get_model_count()
    ref_model_count = gro_file.get_structure().stack_depth()
    assert test_model_count == ref_model_count


@pytest.mark.parametrize(
    "path, model",
    itertools.product(glob.glob(join(data_dir("structure"), "*.gro")), [None, 1, -1]),
)
def test_array_conversion(path, model):
    gro_file = gro.GROFile.read(path)
    array1 = gro_file.get_structure(model=model)
    gro_file = gro.GROFile()
    gro_file.set_structure(array1)
    array2 = gro_file.get_structure(model=model)
    if array1.box is not None:
        assert np.allclose(array1.box, array2.box)
    assert array1.bonds == array2.bonds
    for category in array1.get_annotation_categories():
        assert (
            array1.get_annotation(category).tolist()
            == array2.get_annotation(category).tolist()
        )
    assert array1.coord.tolist() == array2.coord.tolist()


@pytest.mark.parametrize(
    "path", glob.glob(join(data_dir("structure"), "[!(waterbox)]*.gro"))
)
def test_pdb_consistency(path):
    pdb_path = splitext(path)[0] + ".pdb"
    pdb_file = pdb.PDBFile.read(pdb_path)
    a1 = pdb_file.get_structure(model=1)
    gro_file = gro.GROFile.read(path)
    a2 = gro_file.get_structure(model=1)

    assert a1.array_length() == a2.array_length()

    for category in ["res_id", "res_name", "atom_name"]:
        assert (
            a1.get_annotation(category).tolist() == a2.get_annotation(category).tolist()
        )

    # Mind rounding errors when converting pdb to gro (A -> nm)
    assert a1.coord.flatten().tolist() == approx(a2.coord.flatten().tolist(), abs=1e-2)


@pytest.mark.parametrize(
    "path, model",
    itertools.product(glob.glob(join(data_dir("structure"), "*.pdb")), [None, 1, -1]),
)
def test_pdb_to_gro(path, model):
    """
    Converting stacks between formats should not change data
    """
    # Read in data
    pdb_file = pdb.PDBFile.read(path)
    try:
        a1 = pdb_file.get_structure(model=model)
    except biotite.InvalidFileError:
        if model is None:
            # The file cannot be parsed into an AtomArrayStack,
            # as the models contain different numbers of atoms
            # -> skip this test case
            return
        else:
            raise

    # Save stack as gro
    temp = TemporaryFile("w+")
    gro_file = gro.GROFile()
    gro_file.set_structure(a1)
    gro_file.write(temp)

    # Reload stack from gro
    temp.seek(0)
    gro_file = gro.GROFile.read(temp)
    temp.close()
    a2 = gro_file.get_structure(model=model)

    assert a1.array_length() == a2.array_length()

    for category in ["res_id", "res_name", "atom_name"]:
        assert (
            a1.get_annotation(category).tolist() == a2.get_annotation(category).tolist()
        )

    # Mind rounding errors when converting pdb to gro (A -> nm)
    assert a1.coord.flatten().tolist() == approx(a2.coord.flatten().tolist(), abs=1e-2)


def test_gro_id_overflow():
    # Create an oversized AtomArray where atom_id > 100000 and res_id > 10000
    num_atoms = 100005
    atoms = array(
        [
            Atom([1, 2, 3], atom_name="CA", element="C", res_name="X", res_id=i + 1)
            for i in range(num_atoms)
        ]
    )
    atoms.box = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Write .gro file
    temp = TemporaryFile("w+")
    gro_file = gro.GROFile()
    gro_file.set_structure(atoms)
    gro_file.write(temp)

    # Read .gro file
    temp.seek(0)
    gro_file = gro.GROFile.read(temp)
    temp.close()
    s = gro_file.get_structure()

    assert s.array_length() == num_atoms


def test_gro_no_box():
    """
    .gro file format requires valid box parameters at the end of each
    model. However, if we read such a file in, the resulting object should not
    need to have an assigned box.
    """

    # Create an AtomArray
    atom = Atom([1, 2, 3], atom_name="CA", element="C", res_name="X", res_id=1)
    atoms = array([atom])

    # Write .gro file
    temp = TemporaryFile("w+")
    gro_file = gro.GROFile()
    gro_file.set_structure(atoms)
    gro_file.write(temp)

    # Read in file
    temp.seek(0)
    gro_file = gro.GROFile.read(temp)
    temp.close()
    s = gro_file.get_structure()

    # Assert no box with 0 dimension
    assert s.box is None
