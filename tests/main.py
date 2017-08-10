# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import unittest

from .structure import structure_suite
from .sequence import sequence_suite

test_suite = unittest.TestSuite()
test_suite.addTest(structure_suite)
test_suite.addTest(sequence_suite)

def run():
    runner = unittest.TextTestRunner()
    runner.run(test_suite)

if __name__ == "__main__":
    run()