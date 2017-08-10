# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

from setuptools import setup, find_packages
import sys

setup(name="Biopython",
      version = "2.0a1",
      description = "A set of general tools for computational biology",
      author = "The Biopython contributors",
      url = "https://github.com/padix-key/biopython2",
      packages = find_packages("src"),
      package_dir = {"":"src"},
      include_package_data = True,
      install_requires = ["requests",
                          "numpy",
                          "scipy",
                          "matplotlib"],
      test_suite = "tests.main.test_suite",
      zip_safe = False
     )