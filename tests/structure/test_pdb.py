# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

import biopython.structure as struc
import biopython.structure.io.pdb as pdb
import biopython.structure.io.pdbx as pdbx
import biopython.database.rcsb as rcsb
import numpy as np
import glob
from os.path import join
from .util import data_dir
import pytest


@pytest.mark.parametrize("path", glob.glob(join(data_dir, "*.pdb")))
def test_array_conversion(path):
    pdb_file = pdb.PDBFile()
    pdb_file.read(path)
    array1 = pdb_file.get_structure()
    pdb_file.set_structure(array1)
    array2 = pdb_file.get_structure()
    assert array1 == array2


pdb_paths = sorted(glob.glob(join(data_dir, "*.pdb")))
cif_paths = sorted(glob.glob(join(data_dir, "*.cif")))
@pytest.mark.parametrize("pdb_path, cif_path",
                          [(pdb_paths[i], cif_paths[i]) for i
                           in range(len(pdb_paths))])
def test_pdbx_consistency(pdb_path, cif_path):
    pdb_file = pdb.PDBFile()
    pdb_file.read(pdb_path)
    a1 = pdb_file.get_structure()
    pdbx_file = pdbx.PDBxFile()
    pdbx_file.read(cif_path)
    a2 = pdbx.get_structure(pdbx_file)
    if len(a2) == 1:
        a2 = a2.get_array(0)
    # Expected fail in res_id
    for category in ["chain_id", "res_name", "hetero",
                     "atom_name", "element"]:
        assert a1.get_annotation(category).tolist() == \
               a1.get_annotation(category).tolist()
    assert a1.coord.tolist() == a2.coord.tolist()

def test_extra_fields():
    path = join(data_dir, "1l2y.cif")
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