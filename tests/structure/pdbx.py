# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import unittest
import biopython.structure as struc
import biopython.structure.io.pdbx as pdbx
import biopython.database.rcsb as rcsb
import numpy as np
import os
import os.path
from .testutil import data_dir
from .testutil import pdb_ids

class PDBxTest(unittest.TestCase):
    
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.files = rcsb.fetch(pdb_ids, "cif", data_dir)
        
    def test_array_conversion(self):
        self._test_conversion(stack=False)
        
    def test_stack_conversion(self):
        self._test_conversion(stack=True)
            
    def _test_conversion(self, stack):
        for path in self.files:
            with self.subTest(id=path[-8:-4]):
                pdbx_file = pdbx.PDBxFile()
                pdbx_file.read(path)
                if stack:
                    array1 = pdbx.get_structure(pdbx_file)
                else:
                    array1 = pdbx.get_structure(pdbx_file, model=1)
                pdbx.set_structure(pdbx_file, array1)
                if stack:
                    array2 = pdbx.get_structure(pdbx_file)
                else:
                    array2 = pdbx.get_structure(pdbx_file, model=1)
                self.assertEqual(array1, array2)
            
        
    