# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import unittest

from .blast import *

loader = unittest.defaultTestLoader
application_suite = unittest.TestSuite()
application_suite.addTest(loader.loadTestsFromTestCase(BlastTest))

def run():
    runner = unittest.TextTestRunner()
    runner.run(structure_suite)

if __name__ == "__main__":
    run()