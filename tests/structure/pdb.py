# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import unittest
import biopython.structure as struc
import biopython.structure.io.pdb as pdb
import biopython.database.rcsb as rcsb
import numpy as np
import os
import os.path
from .testutil import data_dir
from .testutil import pdb_ids

class PDBTest(unittest.TestCase):
    
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.files = rcsb.fetch(pdb_ids, "pdb", data_dir)
            
    def test_array_conversion(self):
        for path in self.files:
            with self.subTest(id=path[-8:-4]):
                pdb_file = pdb.PDBFile()
                pdb_file.read(path)
                array1 = pdb_file.get_structure()
                pdb_file.set_structure(array1)
                array2 = pdb_file.get_structure()
                self.assertEqual(array1, array2)