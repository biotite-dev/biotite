# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import biotite.structure as struc
import biotite.structure.io.gro as gro
import biotite.structure.io.pdb as pdb
from biotite.structure.atoms import AtomArray as AtomArray
import itertools
import numpy as np
import glob
from os.path import join, basename
from .util import data_dir
import pytest


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
    gro_file = gro.GROFile()
    gro_file.read(gro_paths[file_index])
    a2 = gro_file.get_structure(gro_file, model=model)
    model = None if is_stack else 1
    pdb_file = pdb.PDBFile()
    pdb_file.read(pdb_paths[file_index])
    a1 = pdb_file.get_structure(model=model)

    assert a1.array_length() == a2.array_length()

    for category in ["res_id", "res_name", "atom_name",
                     "element"]:
        assert a1.get_annotation(category).tolist() == \
               a2.get_annotation(category).tolist()


    # mind rounding errors when converting pdb to gros (A -> nm).
    assert False not in np.isclose(a1.coord, a2.coord, atol=0.01)



