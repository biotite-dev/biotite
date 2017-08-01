# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

from setuptools import setup
import sys

setup(name='Biopython',
      version = '2.0a1',
      description = 'Python Distribution Utilities',
      author = "The Biopython contributors",
      url = "https://github.com/padix-key/biopython2",
      packages = ["biopython",
                  "biopython.structure",
                  "biopython.structure.io",
                  "biopython.structure.io.npz",
                  "biopython.structure.io.pdb",
                  "biopython.structure.io.pdbx",
                  "biopython.database",
                  "biopython.database.rcsb"],
      install_requires = ["requests",
                          "numpy",
                          "scipy",
                          "matplotlib"],
      test_suite="tests.test.test_suite"
     )