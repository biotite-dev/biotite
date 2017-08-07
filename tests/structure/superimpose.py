# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import unittest
import biopython.structure as struc
import biopython.structure.io.pdbx as pdbx
import biopython.database.rcsb as rcsb
import biopython.structure as struc
import numpy as np
import os
import os.path
from .testutil import data_dir
from .testutil import pdb_ids

class SuperimposeTest(unittest.TestCase):
    
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
        self.files = rcsb.fetch(pdb_ids, "cif", data_dir)
            
    def test_superimposition(self):
        for path in self.files:
            with self.subTest(id=path[-8:-4]):
                pdbx_file = pdbx.PDBxFile()
                pdbx_file.read(path)
                fixed = pdbx.get_structure(pdbx_file, model=1)
                mobile = fixed.copy()
                mobile = struc.rotate(mobile, (1,2,3))
                mobile = struc.translate(mobile, (1,2,3))
                fitted, transformtion = struc.superimpose(fixed, mobile, False)
                self.assertAlmostEqual(struc.rmsd(fixed, fitted), 0)