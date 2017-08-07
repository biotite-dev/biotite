# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import unittest

from .atoms import *
from .superimpose import *
from .pdbx import *
from .pdb import *

loader = unittest.defaultTestLoader
structure_suite = unittest.TestSuite()
structure_suite.addTest(loader.loadTestsFromTestCase(AtomTypeTest))
structure_suite.addTest(loader.loadTestsFromTestCase(PDBxTest))
structure_suite.addTest(loader.loadTestsFromTestCase(PDBTest))
structure_suite.addTest(loader.loadTestsFromTestCase(SuperimposeTest))

def run():
    runner = unittest.TextTestRunner()
    runner.run(structure_suite)

if __name__ == "__main__":
    run()