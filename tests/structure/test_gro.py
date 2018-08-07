# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite
import biotite.structure.io.gro as gro
import biotite.structure.io.pdb as pdb
import itertools
import numpy as np
import glob
from os.path import join, basename
from .util import data_dir
import pytest
from pytest import approx
import tempfile


@pytest.mark.parametrize("path, is_stack", itertools.product(
                            glob.glob(join(data_dir, "*.gro")),
                            [False, True])
                        )
def test_array_conversion(path, is_stack):
    model = None if is_stack else 1
    gro_file = gro.GROFile()
    gro_file.read(path)
    array1 = gro_file.get_structure(model=model)
    gro_file.set_structure(array1)
    array2 = gro_file.get_structure(model=model)
    assert array1 == array2


pdb_paths = sorted(glob.glob(join(data_dir, "*.pdb")))
gro_paths = sorted(glob.glob(join(data_dir, "*.gro")))
@pytest.mark.parametrize("file_index, is_stack", itertools.product(
                          [i for i in range(len(pdb_paths))],
                          [False, True])
                        )
def test_pdb_consistency(file_index, is_stack):
    print("ID:", basename(gro_paths[file_index])[:-4], "stack:", is_stack)
    model = None if is_stack else 1
    pdb_file = pdb.PDBFile()
    pdb_file.read(pdb_paths[file_index])
    a1 = pdb_file.get_structure(model=model)
    gro_file = gro.GROFile()
    gro_file.read(gro_paths[file_index])
    a2 = gro_file.get_structure(model=model)

    assert a1.array_length() == a2.array_length()

    for category in ["res_id", "res_name", "atom_name"]:
        assert a1.get_annotation(category).tolist() == \
               a2.get_annotation(category).tolist()

    # Mind rounding errors when converting pdb to gro (A -> nm)
    assert a1.coord.tolist() == approx(a2.coord.tolist(), abs=1e-2)

@pytest.mark.parametrize("file_index, is_stack", itertools.product(
                          [i for i in range(len(pdb_paths))],
                          [False, True])
                        )
def test_pdb_to_gro(file_index, is_stack):
    # converting stacks between formats should not change data
    print("ID:", basename(pdb_paths[file_index])[:-4], "stack:", is_stack)
    model = None if is_stack else 1

    # Read in data
    pdb_file = pdb.PDBFile()
    pdb_file.read(pdb_paths[file_index])
    a1 = pdb_file.get_structure(model=model)

    # Save stack as gro
    tmp = tempfile.NamedTemporaryFile(suffix=".gro").name
    gro_file = gro.GROFile()
    gro_file.set_structure(a1)
    gro_file.write(tmp)

    # Reload stack from gro
    gro_file = gro.GROFile()
    gro_file.read(tmp)
    a2 = gro_file.get_structure(model=model)

    assert a1.array_length() == a2.array_length()

    for category in ["res_id", "res_name", "atom_name"]:
        assert a1.get_annotation(category).tolist() == \
               a2.get_annotation(category).tolist()

    # Mind rounding errors when converting pdb to gro (A -> nm)
    assert a1.coord.tolist() == approx(a2.coord.tolist(), abs=1e-2)



