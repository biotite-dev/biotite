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
from biotite.structure.atoms import AtomArray
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
    array1 = pdb_file.get_structure(model=model)
    pdb_file = pdb.PDBFile()
    pdb_file.set_structure(array1)
    array2 = pdb_file.get_structure(model=model)
    assert array1 == array2


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


