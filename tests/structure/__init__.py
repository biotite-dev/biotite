# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import unittest

from .atoms import *
from .pdbx import *

loader = unittest.defaultTestLoader
structure_suite = unittest.TestSuite()
structure_suite.addTest(loader.loadTestsFromTestCase(AtomTypeTest))
structure_suite.addTest(loader.loadTestsFromTestCase(PDBxTest))

def run():
    runner = unittest.TextTestRunner()
    runner.run(structure_suite)

if __name__ == "__main__":
    run()