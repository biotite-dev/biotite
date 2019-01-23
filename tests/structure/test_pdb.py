# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
import numpy as np
import glob
from os.path import join, splitext
import pytest
from pytest import approx
import biotite
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
from biotite.structure.atoms import AtomArray, array, Atom
import biotite.structure.io as io
from .util import data_dir


@pytest.mark.parametrize(
    "path, single_model",
    itertools.product(
        glob.glob(join(data_dir, "*.pdb")),
        [False, True]
    )
)
def test_array_conversion(path, single_model):
    model = 1 if single_model else None
    pdb_file = pdb.PDBFile()
    pdb_file.read(path)
    # Test also the thin wrapper around the methods
    # 'get_structure()' and 'set_structure()'
    array1 = pdb.get_structure(pdb_file, model=model)
    pdb_file = pdb.PDBFile()
    pdb.set_structure(pdb_file, array1)
    array2 = pdb.get_structure(pdb_file, model=model)
    if array1.box is not None:
        assert np.allclose(array1.box, array2.box)
    assert array1.bonds == array2.bonds
    for category in array1.get_annotation_categories():
        assert array1.get_annotation(category).tolist() == \
               array2.get_annotation(category).tolist()
    assert array1.coord.tolist() == array2.coord.tolist()


@pytest.mark.parametrize(
    "path, single_model",
    itertools.product(
        glob.glob(join(data_dir, "*.pdb")),
        [False, True]
    )
)
def test_pdbx_consistency(path, single_model):
    model = 1 if single_model else None
    cif_path = splitext(path)[0] + ".cif"
    pdb_file = pdb.PDBFile()
    pdb_file.read(path)
    a1 = pdb_file.get_structure(model=model)
    pdbx_file = pdbx.PDBxFile()
    pdbx_file.read(cif_path)
    a2 = pdbx.get_structure(pdbx_file, model=model)
    if a2.box is not None:
        assert np.allclose(a1.box, a2.box)
    assert a1.bonds == a2.bonds
    for category in a1.get_annotation_categories():
        assert a1.get_annotation(category).tolist() == \
               a2.get_annotation(category).tolist()
    assert a1.coord.tolist() == a2.coord.tolist()

def test_extra_fields():
    path = join(data_dir, "1l2y.pdb")
    pdb_file = pdb.PDBFile()
    pdb_file.read(path)
    stack1 = pdb_file.get_structure(extra_fields=["atom_id","b_factor",
                                                  "occupancy","charge"])
    pdb_file.set_structure(stack1)
    stack2 = pdb_file.get_structure(extra_fields=["atom_id","b_factor",
                                                  "occupancy","charge"])
    assert stack1.atom_id.tolist() == stack2.atom_id.tolist()
    assert stack1.b_factor.tolist() == stack2.b_factor.tolist()
    assert stack1.occupancy.tolist() == stack2.occupancy.tolist()
    assert stack1.charge.tolist() == stack2.charge.tolist()
    assert stack1 == stack2


@pytest.mark.filterwarnings("ignore")
def test_guess_elements():
    # read valid pdb file
    path = join(data_dir, "1l2y.pdb")
    pdb_file = pdb.PDBFile()
    pdb_file.read(path)
    stack = pdb_file.get_structure()

    # remove all elements
    removed_stack = stack.copy()
    removed_stack.element[:] = ''

    # save stack without elements to tmp file
    tmp_file_name = biotite.temp_file(".pdb")
    tmp_pdb_file = pdb.PDBFile()
    tmp_pdb_file.set_structure(removed_stack)
    tmp_pdb_file.write(tmp_file_name)

    # read new stack from file with guessed elements
    guessed_pdb_file = pdb.PDBFile()
    guessed_pdb_file.read(tmp_file_name)
    guessed_stack = guessed_pdb_file.get_structure()

    assert guessed_stack.element.tolist() == stack.element.tolist()


@pytest.mark.parametrize(
    "path, single_model",
    itertools.product(
        glob.glob(join(data_dir, "*.pdb")),
        [False, True]
    )
)
def test_box_shape(path, single_model):
    model = 1 if single_model else None
    pdb_file = pdb.PDBFile()
    pdb_file.read(path)
    a = pdb_file.get_structure(model=model)

    if isinstance(a, AtomArray):
        expected_box_dim = (3, 3)
    else:
        expected_box_dim = (len(a), 3, 3)
    assert expected_box_dim == a.box.shape


def test_box_parsing():
    path = join(data_dir, "1igy.pdb")
    pdb_file = pdb.PDBFile()
    pdb_file.read(path)
    a = pdb_file.get_structure()
    expected_box = np.array([[
        [66.65,   0.00, 0.00],
        [0.00,  190.66, 0.00],
        [-24.59,  0.00, 68.84]
    ]])

    assert expected_box.flatten().tolist() \
           == approx(a.box.flatten().tolist(), abs=1e-2)


def test_atoms_overflow():
    # Create a stack > 100k atoms
    atoms = [Atom([1,2,3]) for i in range(100000)]
    a = array(atoms)
    a.res_id = np.array([1] * 100000)
    a.atom_name = np.array(['CA'] * 100000)
    
    # Write stack to pdb file and make sure a warning is thrown
    with pytest.warns(UserWarning):
        tmp_file_name = biotite.temp_file(".pdb")
        tmp_pdb_file = pdb.PDBFile()
        tmp_pdb_file.set_structure(a)
        tmp_pdb_file.write(tmp_file_name)

    # Assert file can be read properly
    a2 = io.load_structure(tmp_file_name)
    assert(a2.array_length() == a.array_length())
    
    # Manually check if the written atom id is correct
    with open(tmp_file_name) as output:
        last_line = output.readlines()[-1]
        atom_id = int(last_line.split()[1])
        assert(atom_id == 1)
