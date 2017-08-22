# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import unittest
import biopython.structure as struc
import biopython.structure.io.pdb as pdb
import biopython.structure.io.pdbx as pdbx
import biopython.database.rcsb as rcsb
import numpy as np
import os
import os.path
from .testutil import data_dir
from .testutil import pdb_ids

class PDBTest(unittest.TestCase):
    
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.pdb_files = rcsb.fetch(pdb_ids, "pdb", data_dir)
        self.pdbx_files = rcsb.fetch(pdb_ids, "cif", data_dir)
            
    def test_array_conversion(self):
        for path in self.pdb_files:
            with self.subTest(id=path[-8:-4]):
                pdb_file = pdb.PDBFile()
                pdb_file.read(path)
                array1 = pdb_file.get_structure()
                pdb_file.set_structure(array1)
                array2 = pdb_file.get_structure()
                self.assertEqual(array1, array2)
    
    def test_pdbx_consistency(self):
        for i in range(len(self.pdb_files)):
            pdb_path = self.pdb_files[i]
            pdbx_path = self.pdbx_files[i]
            with self.subTest(id=pdb_path[-8:-4]):
                pdb_file = pdb.PDBFile()
                pdb_file.read(pdb_path)
                a1 = pdb_file.get_structure()
                pdbx_file = pdbx.PDBxFile()
                pdbx_file.read(pdbx_path)
                a2 = pdbx.get_structure(pdbx_file)
                if len(a2) == 1:
                    a2 = a2.get_array(0)
                # Expected fail in res_id
                for category in ["chain_id", "res_name", "hetero",
                                 "atom_name", "element"]:
                    self.assertEqual(list(a1.get_annotation(category)),
                                     list(a2.get_annotation(category)),
                                     msg="Fail in " + category)
                self.assertTrue(np.array_equal(a1.coord, a2.coord))