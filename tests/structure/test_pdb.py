# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import itertools
import numpy as np
import glob
from os.path import join, basename
from .util import data_dir
import pytest


@pytest.mark.parametrize("path, is_stack", itertools.product(
                            glob.glob(join(data_dir, "*.pdb")),
                            [False, True])
                        )
def test_array_conversion(path, is_stack):
    model = None if is_stack else 1
    pdb_file = pdb.PDBFile()
    pdb_file.read(path)
    array1 = pdb_file.get_structure(model=model)
    pdb_file.set_structure(array1)
    array2 = pdb_file.get_structure(model=model)
    assert array1 == array2


pdb_paths = sorted(glob.glob(join(data_dir, "*.pdb")))
cif_paths  = sorted(glob.glob(join(data_dir, "*.cif" )))
@pytest.mark.parametrize("file_index, is_stack", itertools.product(
                          [i for i in range(len(pdb_paths))],
                          [False, True])
                        )
def test_pdbx_consistency(file_index, is_stack):
    print("ID:", basename(cif_paths[file_index])[:-4])
    model = None if is_stack else 1
    pdb_file = pdb.PDBFile()
    pdb_file.read(pdb_paths[file_index])
    a1 = pdb_file.get_structure(model=model)
    pdbx_file = pdbx.PDBxFile()
    pdbx_file.read(cif_paths[file_index])
    a2 = pdbx.get_structure(pdbx_file, model=model)
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
    import tempfile
    from copy import deepcopy

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

