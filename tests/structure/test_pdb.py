# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

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


pdb_paths = glob.glob(join(data_dir, "*.pdb"))
cif_paths = glob.glob(join(data_dir, "*.cif"))
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